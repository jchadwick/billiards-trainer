"""YOLO-based object detection for billiards vision system.

This module provides YOLOv8-based detection for balls, cue sticks, and table elements.
Implements FR-VIS-056 through FR-VIS-060 for deep learning detection capabilities.

Features:
- YOLOv8 model support (.pt and .onnx formats)
- Ball detection with type classification
- Cue stick detection with orientation
- Table element detection
- Automatic fallback to OpenCV when model unavailable
- GPU/CPU device selection
- Confidence and NMS threshold configuration

The detector returns raw detections that are converted to Ball/CueStick objects
by adapter classes in the vision pipeline.
"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ModelValidationError(Exception):
    """Raised when model validation fails."""

    pass


class ModelFormat(Enum):
    """Supported model formats."""

    PYTORCH = "pt"
    ONNX = "onnx"
    TENSORRT = "engine"


class BallClass(Enum):
    """YOLO class IDs for ball types.

    These map to the class IDs in the trained YOLO model.
    The actual mapping depends on how the model was trained.
    """

    CUE_BALL = 0  # White cue ball
    BALL_1 = 1  # Yellow solid
    BALL_2 = 2  # Blue solid
    BALL_3 = 3  # Red solid
    BALL_4 = 4  # Purple solid
    BALL_5 = 5  # Orange solid
    BALL_6 = 6  # Green solid
    BALL_7 = 7  # Maroon/brown solid
    BALL_8 = 8  # Black 8-ball
    BALL_9 = 9  # Yellow stripe
    BALL_10 = 10  # Blue stripe
    BALL_11 = 11  # Red stripe
    BALL_12 = 12  # Purple stripe
    BALL_13 = 13  # Orange stripe
    BALL_14 = 14  # Green stripe
    BALL_15 = 15  # Maroon/brown stripe
    # Additional classes for other objects
    CUE_STICK = 16  # Cue stick
    TABLE = 17  # Table surface
    POCKET = 18  # Table pocket


@dataclass
class Detection:
    """Raw detection from YOLO model.

    This is the intermediate format before conversion to Ball/CueStick objects.
    """

    class_id: int  # YOLO class ID
    class_name: str  # Human-readable class name
    confidence: float  # Detection confidence (0.0-1.0)
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) bounding box
    center: tuple[float, float]  # Center point (x, y)
    width: float  # Bounding box width
    height: float  # Bounding box height
    angle: float = 0.0  # Rotation angle (for oriented objects like cue sticks)
    mask: Optional[NDArray[np.uint8]] = None  # Segmentation mask if available


@dataclass
class TableElements:
    """Detected table elements from YOLO.

    Represents table surface, pockets, and other structural elements.
    """

    table_bbox: Optional[tuple[float, float, float, float]] = None
    table_confidence: float = 0.0
    pockets: list[Detection] = None
    rails: list[Detection] = None

    def __post_init__(self):
        """Initialize empty lists."""
        if self.pockets is None:
            self.pockets = []
        if self.rails is None:
            self.rails = []


class YOLODetector:
    """YOLO-based object detector for billiards vision system.

    Provides high-performance deep learning detection for balls, cue sticks,
    and table elements using YOLOv8 models.

    Supports both PyTorch (.pt) and ONNX (.onnx) model formats, with automatic
    device selection (GPU/CPU) and graceful fallback handling.

    Usage:
        detector = YOLODetector(model_path="models/billiards_yolo.pt")
        balls = detector.detect_balls(frame)
        cue = detector.detect_cue(frame)
        table = detector.detect_table_elements(frame)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        confidence: float = 0.4,
        nms_threshold: float = 0.45,
        auto_fallback: bool = True,
    ) -> None:
        """Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model file (.pt or .onnx). If None, detector
                       will operate in fallback mode without YOLO.
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            confidence: Minimum confidence threshold (0.0-1.0)
            nms_threshold: Non-maximum suppression IoU threshold
            auto_fallback: Automatically fall back to None model if loading fails

        Raises:
            FileNotFoundError: If model_path is provided but file doesn't exist
            RuntimeError: If model loading fails and auto_fallback is False
        """
        self.model_path = model_path
        self.device = device
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.auto_fallback = auto_fallback

        # Model state
        self.model: Optional[Any] = None
        self.model_format: Optional[ModelFormat] = None
        self.model_loaded = False

        # Thread safety for model hot-swapping
        self._model_lock = threading.RLock()

        # Class mapping
        self._init_class_mapping()

        # Statistics
        self.stats = {
            "total_inferences": 0,
            "total_detections": 0,
            "avg_inference_time": 0.0,
            "fallback_mode": False,
        }

        # Load model if path provided
        if model_path is not None:
            try:
                self._load_model(model_path)
            except Exception as e:
                if auto_fallback:
                    logger.warning(
                        f"Failed to load YOLO model from {model_path}: {e}. "
                        "Operating in fallback mode."
                    )
                    self.stats["fallback_mode"] = True
                else:
                    raise RuntimeError(f"Failed to load YOLO model: {e}") from e
        else:
            logger.info("No model path provided, operating in fallback mode")
            self.stats["fallback_mode"] = True

    def _init_class_mapping(self) -> None:
        """Initialize class ID to name mapping."""
        self.class_names = {
            0: "cue_ball",
            1: "ball_1",
            2: "ball_2",
            3: "ball_3",
            4: "ball_4",
            5: "ball_5",
            6: "ball_6",
            7: "ball_7",
            8: "ball_8",
            9: "ball_9",
            10: "ball_10",
            11: "ball_11",
            12: "ball_12",
            13: "ball_13",
            14: "ball_14",
            15: "ball_15",
            16: "cue_stick",
            17: "table",
            18: "pocket",
        }

        # Reverse mapping
        self.name_to_class = {v: k for k, v in self.class_names.items()}

    def _load_model(self, model_path: str) -> None:
        """Load YOLO model from file.

        Args:
            model_path: Path to model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            ImportError: If ultralytics package not installed
            RuntimeError: If model loading fails
        """
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Determine model format from extension
        suffix = path.suffix.lower()
        if suffix == ".pt":
            self.model_format = ModelFormat.PYTORCH
        elif suffix == ".onnx":
            self.model_format = ModelFormat.ONNX
        elif suffix == ".engine":
            self.model_format = ModelFormat.TENSORRT
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

        try:
            # Import ultralytics YOLO
            from ultralytics import YOLO

            logger.info(
                f"Loading YOLO model from {model_path} (format: {self.model_format.value})"
            )

            # Load model
            self.model = YOLO(str(path))

            # Set device
            if self.device != "cpu":
                # Ultralytics YOLO will auto-detect GPU availability
                logger.info(f"Attempting to use device: {self.device}")

            self.model_loaded = True
            logger.info(f"YOLO model loaded successfully on device: {self.device}")

        except ImportError as e:
            raise ImportError(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}") from e

    @staticmethod
    def validate_onnx_model(model_path: str) -> dict[str, Any]:
        """Validate ONNX model file and extract metadata.

        Args:
            model_path: Path to ONNX model file

        Returns:
            Dictionary with model metadata:
                - valid: bool - whether model is valid
                - input_shape: tuple - model input shape (if available)
                - output_shape: tuple - model output shape (if available)
                - num_classes: int - number of output classes (if detectable)
                - opset_version: int - ONNX opset version
                - producer: str - model producer/framework
                - error: str - error message if validation failed

        Raises:
            ModelValidationError: If model is invalid or cannot be validated
        """
        path = Path(model_path)

        if not path.exists():
            raise ModelValidationError(f"Model file not found: {model_path}")

        if path.suffix.lower() != ".onnx":
            raise ModelValidationError(f"File is not an ONNX model: {model_path}")

        try:
            # Try to import onnx for validation
            try:
                import onnx

                has_onnx = True
            except ImportError:
                logger.warning(
                    "onnx package not installed, performing basic validation only"
                )
                has_onnx = False

            metadata = {
                "valid": False,
                "file_size": path.stat().st_size,
                "error": None,
            }

            if has_onnx:
                # Load and validate ONNX model
                try:
                    model = onnx.load(str(path))

                    # Check model validity
                    onnx.checker.check_model(model)
                    metadata["valid"] = True

                    # Extract metadata
                    metadata["opset_version"] = (
                        model.opset_import[0].version if model.opset_import else None
                    )
                    metadata["producer"] = (
                        model.producer_name
                        if hasattr(model, "producer_name")
                        else "unknown"
                    )
                    metadata["producer_version"] = (
                        model.producer_version
                        if hasattr(model, "producer_version")
                        else None
                    )

                    # Extract input/output shapes
                    if model.graph.input:
                        input_shape = []
                        for dim in model.graph.input[0].type.tensor_type.shape.dim:
                            if dim.dim_value:
                                input_shape.append(dim.dim_value)
                            else:
                                input_shape.append(-1)  # Dynamic dimension
                        metadata["input_shape"] = (
                            tuple(input_shape) if input_shape else None
                        )
                        metadata["input_name"] = model.graph.input[0].name

                    if model.graph.output:
                        output_shape = []
                        for dim in model.graph.output[0].type.tensor_type.shape.dim:
                            if dim.dim_value:
                                output_shape.append(dim.dim_value)
                            else:
                                output_shape.append(-1)  # Dynamic dimension
                        metadata["output_shape"] = (
                            tuple(output_shape) if output_shape else None
                        )
                        metadata["output_name"] = model.graph.output[0].name

                        # Try to infer number of classes from output shape
                        # YOLO models typically have output shape like (batch, num_predictions, 4+num_classes)
                        # or (batch, num_classes+5, grid, grid) depending on version
                        if output_shape and len(output_shape) >= 2:
                            # This is a heuristic - may not work for all YOLO variants
                            metadata["num_classes"] = (
                                output_shape[-1] - 5
                            )  # Common YOLO format

                    logger.info(f"ONNX model validation successful: {model_path}")

                except onnx.checker.ValidationError as e:
                    metadata["valid"] = False
                    metadata["error"] = f"ONNX validation failed: {str(e)}"
                    raise ModelValidationError(metadata["error"])
                except Exception as e:
                    metadata["valid"] = False
                    metadata["error"] = f"Failed to parse ONNX model: {str(e)}"
                    raise ModelValidationError(metadata["error"])
            else:
                # Basic validation without onnx package
                # Just check if it looks like an ONNX file
                with open(path, "rb") as f:
                    # ONNX files start with specific protobuf header
                    header = f.read(16)
                    if len(header) < 4:
                        raise ModelValidationError(
                            "File is too small to be a valid ONNX model"
                        )

                    # Very basic check - just verify it's not empty and has reasonable size
                    if metadata["file_size"] < 1000:
                        raise ModelValidationError(
                            "File is too small to be a valid ONNX model"
                        )

                    metadata["valid"] = True
                    metadata["warning"] = (
                        "Full validation requires 'onnx' package - install with: pip install onnx"
                    )

            return metadata

        except ModelValidationError:
            raise
        except Exception as e:
            raise ModelValidationError(
                f"Unexpected error validating model: {str(e)}"
            ) from e

    @staticmethod
    def test_model_inference(model_path: str, device: str = "cpu") -> dict[str, Any]:
        """Test if model can be loaded and run inference.

        Args:
            model_path: Path to model file (.pt or .onnx)
            device: Device to test on ('cpu', 'cuda', 'mps')

        Returns:
            Dictionary with test results:
                - success: bool - whether test succeeded
                - inference_time: float - time in milliseconds
                - error: str - error message if test failed
                - output_shape: tuple - output shape from test inference

        Raises:
            ModelValidationError: If test fails
        """
        import time

        try:
            # Create temporary detector
            detector = YOLODetector(
                model_path=model_path,
                device=device,
                auto_fallback=False,
            )

            # Create test image (640x640 is standard YOLO input)
            test_frame = np.zeros((640, 640, 3), dtype=np.uint8)

            # Run test inference
            start_time = time.time()
            detections = detector._run_inference(test_frame)
            inference_time = (time.time() - start_time) * 1000

            return {
                "success": True,
                "inference_time": inference_time,
                "num_detections": len(detections),
                "model_loaded": detector.model_loaded,
                "device": device,
                "error": None,
            }

        except Exception as e:
            error_msg = f"Model inference test failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "inference_time": None,
                "error": error_msg,
            }

    def detect_balls(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Detect all balls in the frame.

        Args:
            frame: Input image in BGR format

        Returns:
            List of ball detections (class IDs 0-15)
        """
        if not self.model_loaded:
            logger.debug("Model not loaded, returning empty ball detections")
            return []

        try:
            # Run inference
            detections = self._run_inference(frame)

            # Filter for ball classes (0-15)
            ball_detections = [det for det in detections if 0 <= det.class_id <= 15]

            return ball_detections

        except Exception as e:
            logger.error(f"Ball detection failed: {e}")
            return []

    def detect_cue(self, frame: NDArray[np.uint8]) -> Optional[Detection]:
        """Detect cue stick in the frame.

        Args:
            frame: Input image in BGR format

        Returns:
            Cue stick detection or None if not detected
        """
        if not self.model_loaded:
            logger.debug("Model not loaded, returning no cue detection")
            return None

        try:
            # Run inference
            detections = self._run_inference(frame)

            # Filter for cue stick class (16)
            cue_detections = [det for det in detections if det.class_id == 16]

            # Return highest confidence cue detection
            if cue_detections:
                cue_detections.sort(key=lambda d: d.confidence, reverse=True)
                cue = cue_detections[0]

                # Calculate angle from bounding box orientation
                # For now, use simple heuristic: if bbox is wider than tall,
                # angle is close to horizontal
                if cue.width > cue.height:
                    # Horizontal-ish
                    cue.angle = 0.0
                else:
                    # Vertical-ish
                    cue.angle = 90.0

                return cue

            return None

        except Exception as e:
            logger.error(f"Cue detection failed: {e}")
            return None

    def detect_table_elements(self, frame: NDArray[np.uint8]) -> TableElements:
        """Detect table surface, pockets, and other structural elements.

        Args:
            frame: Input image in BGR format

        Returns:
            TableElements with detected components
        """
        if not self.model_loaded:
            logger.debug("Model not loaded, returning empty table elements")
            return TableElements()

        try:
            # Run inference
            detections = self._run_inference(frame)

            # Filter for table class (17)
            table_detections = [det for det in detections if det.class_id == 17]

            # Filter for pocket class (18)
            pocket_detections = [det for det in detections if det.class_id == 18]

            # Build TableElements
            elements = TableElements()

            if table_detections:
                # Use highest confidence table detection
                table_detections.sort(key=lambda d: d.confidence, reverse=True)
                table = table_detections[0]
                elements.table_bbox = table.bbox
                elements.table_confidence = table.confidence

            elements.pockets = pocket_detections

            return elements

        except Exception as e:
            logger.error(f"Table element detection failed: {e}")
            return TableElements()

    def _run_inference(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run YOLO inference on frame and convert results to Detection objects.

        Args:
            frame: Input image in BGR format

        Returns:
            List of all detections
        """
        if not self.model_loaded:
            return []

        import time

        start_time = time.time()

        try:
            # Acquire lock for thread-safe model access during inference
            with self._model_lock:
                if not self.model_loaded or self.model is None:
                    return []

                # Run YOLO inference
                # ultralytics YOLO expects BGR format (OpenCV standard)
                results = self.model(
                    frame,
                    conf=self.confidence,
                    iou=self.nms_threshold,
                    verbose=False,
                )

            # Parse results (outside lock - parsing doesn't need model access)
            detections = []

            # Results is a list (batch processing), we use single image
            if results and len(results) > 0:
                result = results[0]

                # Extract boxes
                if hasattr(result, "boxes") and result.boxes is not None:
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        # Get box data
                        box = boxes[i]

                        # Coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Class and confidence
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())

                        # Calculate center and dimensions
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1

                        # Get class name
                        class_name = self.class_names.get(
                            class_id, f"unknown_{class_id}"
                        )

                        # Create detection
                        detection = Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            center=(center_x, center_y),
                            width=width,
                            height=height,
                        )

                        detections.append(detection)

            # Update statistics
            inference_time = (time.time() - start_time) * 1000  # ms
            self.stats["total_inferences"] += 1
            self.stats["total_detections"] += len(detections)

            # Update average inference time (exponential moving average)
            alpha = 0.1
            self.stats["avg_inference_time"] = (
                alpha * inference_time + (1 - alpha) * self.stats["avg_inference_time"]
            )

            logger.debug(
                f"YOLO inference: {len(detections)} detections in {inference_time:.1f}ms"
            )

            return detections

        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            return []

    def is_available(self) -> bool:
        """Check if YOLO model is loaded and available.

        Returns:
            True if model is loaded and ready for inference
        """
        return self.model_loaded

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_path": str(self.model_path) if self.model_path else None,
            "model_format": self.model_format.value if self.model_format else None,
            "model_loaded": self.model_loaded,
            "device": self.device,
            "confidence_threshold": self.confidence,
            "nms_threshold": self.nms_threshold,
            "fallback_mode": self.stats["fallback_mode"],
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get detection statistics.

        Returns:
            Dictionary with statistics
        """
        return self.stats.copy()

    def update_thresholds(
        self, confidence: Optional[float] = None, nms: Optional[float] = None
    ) -> None:
        """Update detection thresholds.

        Args:
            confidence: New confidence threshold (0.0-1.0)
            nms: New NMS IoU threshold (0.0-1.0)
        """
        if confidence is not None:
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("Confidence must be between 0.0 and 1.0")
            self.confidence = confidence
            logger.info(f"Updated confidence threshold to {confidence}")

        if nms is not None:
            if not 0.0 <= nms <= 1.0:
                raise ValueError("NMS threshold must be between 0.0 and 1.0")
            self.nms_threshold = nms
            logger.info(f"Updated NMS threshold to {nms}")

    def _validate_model_file(self, model_path: str) -> bool:
        """Validate model file before loading.

        Args:
            model_path: Path to model file

        Returns:
            True if model is valid

        Raises:
            ModelValidationError: If model validation fails
        """
        path = Path(model_path)

        # Check file exists
        if not path.exists():
            raise ModelValidationError(f"Model file not found: {model_path}")

        # Check file extension
        suffix = path.suffix.lower()
        valid_extensions = [".pt", ".onnx", ".engine"]
        if suffix not in valid_extensions:
            raise ModelValidationError(
                f"Invalid model format: {suffix}. Expected one of {valid_extensions}"
            )

        # Check file is readable and has content
        try:
            file_size = path.stat().st_size
            if file_size == 0:
                raise ModelValidationError(f"Model file is empty: {model_path}")

            # Minimum reasonable size for a YOLO model (100KB)
            if file_size < 100 * 1024:
                raise ModelValidationError(
                    f"Model file suspiciously small ({file_size} bytes): {model_path}"
                )

            logger.info(
                f"Model validation passed: {model_path} ({file_size / (1024*1024):.2f} MB)"
            )
            return True

        except OSError as e:
            raise ModelValidationError(f"Cannot access model file: {e}") from e

    def reload_model(self, model_path: Optional[str] = None) -> bool:
        """Reload model with thread-safe hot-swapping and validation.

        Implements FR-VIS-058: Support model hot-swapping without system restart.

        This method provides:
        - Thread-safe model swapping using locks
        - Model validation before loading
        - Fallback to previous model if new model fails
        - Comprehensive logging of model changes

        Args:
            model_path: New model path, or None to reload current model

        Returns:
            True if model loaded successfully, False otherwise

        Raises:
            ModelValidationError: If model validation fails (only when auto_fallback=False)
        """
        # Determine which model path to use
        new_model_path = model_path if model_path is not None else self.model_path

        if new_model_path is None:
            logger.error("No model path specified for reload")
            return False

        logger.info(f"Initiating model reload from: {new_model_path}")

        # Acquire lock for thread-safe model swapping
        with self._model_lock:
            # Save current model state for fallback
            previous_model = self.model
            previous_model_path = self.model_path
            previous_model_format = self.model_format
            previous_model_loaded = self.model_loaded

            try:
                # Validate new model before attempting to load
                logger.info(f"Validating model: {new_model_path}")
                self._validate_model_file(str(new_model_path))

                # Clear current model
                logger.info("Clearing current model for reload")
                self.model = None
                self.model_loaded = False
                self.model_path = new_model_path

                # Load new model
                logger.info(f"Loading new model from: {new_model_path}")
                self._load_model(str(new_model_path))

                # Verify model loaded successfully
                if not self.model_loaded or self.model is None:
                    raise RuntimeError("Model loaded but not marked as available")

                # Test model with a dummy inference to ensure it works
                logger.info("Testing new model with dummy inference")
                test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model(test_frame, conf=0.5, verbose=False)

                # Success! Log the change
                logger.info(
                    f"Model reloaded successfully: {previous_model_path} -> {new_model_path}"
                )
                logger.info(
                    f"Model format: {self.model_format.value if self.model_format else 'unknown'}"
                )

                # Clear fallback mode if we were in it
                if self.stats["fallback_mode"]:
                    logger.info("Exiting fallback mode - model loaded successfully")
                    self.stats["fallback_mode"] = False

                return True

            except Exception as e:
                logger.error(f"Model reload failed: {e}", exc_info=True)

                # Attempt to restore previous model
                logger.warning("Attempting to restore previous model")
                self.model = previous_model
                self.model_path = previous_model_path
                self.model_format = previous_model_format
                self.model_loaded = previous_model_loaded

                # Verify previous model still works
                if previous_model is not None:
                    try:
                        test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                        _ = previous_model(test_frame, conf=0.5, verbose=False)
                        logger.info("Successfully restored previous model")
                    except Exception as restore_error:
                        logger.error(
                            f"Failed to restore previous model: {restore_error}",
                            exc_info=True,
                        )
                        self.model = None
                        self.model_loaded = False
                        logger.error(
                            "Model reload failed and previous model is unusable"
                        )

                # Handle based on auto_fallback setting
                if self.auto_fallback:
                    self.stats["fallback_mode"] = True
                    logger.warning(
                        "Operating in fallback mode due to model reload failure"
                    )
                    return False
                else:
                    raise ModelValidationError(f"Model reload failed: {e}") from e

    def visualize_detections(
        self,
        frame: NDArray[np.uint8],
        detections: list[Detection],
        show_labels: bool = True,
        show_confidence: bool = True,
    ) -> NDArray[np.uint8]:
        """Visualize detections on frame.

        Args:
            frame: Input image
            detections: List of detections to visualize
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores

        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()

        # Color map for different object types
        colors = {
            "ball": (0, 255, 0),  # Green for balls
            "cue": (255, 0, 0),  # Blue for cue stick
            "table": (0, 255, 255),  # Yellow for table
            "pocket": (0, 0, 255),  # Red for pockets
        }

        for det in detections:
            # Determine color based on class
            if 0 <= det.class_id <= 15:
                color = colors["ball"]
            elif det.class_id == 16:
                color = colors["cue"]
            elif det.class_id == 17:
                color = colors["table"]
            elif det.class_id == 18:
                color = colors["pocket"]
            else:
                color = (128, 128, 128)  # Gray for unknown

            # Draw bounding box
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # Draw center point
            cx, cy = (int(v) for v in det.center)
            cv2.circle(vis_frame, (cx, cy), 4, color, -1)

            # Draw label
            if show_labels or show_confidence:
                label_parts = []
                if show_labels:
                    label_parts.append(det.class_name)
                if show_confidence:
                    label_parts.append(f"{det.confidence:.2f}")

                label = " ".join(label_parts)

                # Text background
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    vis_frame,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    color,
                    -1,
                )

                # Text
                cv2.putText(
                    vis_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        return vis_frame


# Convenience functions for common operations


def create_detector(
    model_path: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> YOLODetector:
    """Create YOLO detector with configuration.

    Args:
        model_path: Path to YOLO model file
        config: Configuration dictionary with keys:
            - device: 'cpu', 'cuda', 'mps'
            - confidence: float 0.0-1.0
            - nms_threshold: float 0.0-1.0
            - auto_fallback: bool

    Returns:
        Configured YOLODetector instance
    """
    if config is None:
        config = {}

    return YOLODetector(
        model_path=model_path,
        device=config.get("device", "cpu"),
        confidence=config.get("confidence", 0.4),
        nms_threshold=config.get("nms_threshold", 0.45),
        auto_fallback=config.get("auto_fallback", True),
    )


def ball_class_to_type(class_id: int) -> tuple[str, Optional[int]]:
    """Convert YOLO ball class ID to ball type and number.

    Args:
        class_id: YOLO class ID (0-15)

    Returns:
        Tuple of (ball_type, ball_number)
        ball_type is one of: "cue", "solid", "stripe", "eight"
        ball_number is 1-15 for numbered balls, None for cue ball
    """
    if class_id == 0:
        return ("cue", None)
    elif class_id == 8:
        return ("eight", 8)
    elif 1 <= class_id <= 7:
        return ("solid", class_id)
    elif 9 <= class_id <= 15:
        return ("stripe", class_id)
    else:
        return ("unknown", None)
