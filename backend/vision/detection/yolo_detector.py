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
    EDGETPU = "tflite"  # Edge TPU TensorFlow Lite


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
        confidence: float = 0.15,
        nms_threshold: float = 0.45,
        auto_fallback: bool = True,
        tpu_device_path: Optional[str] = None,
        enable_opencv_classification: bool = True,
        min_ball_size: int = 20,
    ) -> None:
        """Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model file (.pt, .onnx, or .tflite). If None, detector
                       will operate in fallback mode without YOLO.
            device: Device to run inference on ('cpu', 'cuda', 'mps', 'tpu')
            confidence: Minimum confidence threshold (0.0-1.0). Default 0.15 works well for
                       billiards ball detection while capturing difficult angles and lighting.
            nms_threshold: Non-maximum suppression IoU threshold
            auto_fallback: Automatically fall back to None model if loading fails
            tpu_device_path: Coral TPU device path (e.g., '/dev/bus/usb/001/002', 'usb', 'pcie', or None for auto)
            enable_opencv_classification: Enable OpenCV-based ball type classification refinement.
                                         Enabled by default as it provides accurate ball number detection
                                         even when YOLO detects balls as generic "ball" class.
            min_ball_size: Minimum ball size in pixels (width or height) to filter out small detections
                          like markers and noise. Default 20px works well for typical camera setups.

        Raises:
            FileNotFoundError: If model_path is provided but file doesn't exist
            RuntimeError: If model loading fails and auto_fallback is False

        Note:
            These defaults (confidence=0.15, enable_opencv_classification=True, min_ball_size=20)
            are based on extensive testing with the VisionModule and video_debugger tool.
            They provide the best balance of detection recall (catching all balls) while
            maintaining precision (filtering out false positives like markers and shadows).
        """
        self.model_path = model_path
        self.device = device
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.auto_fallback = auto_fallback
        self.tpu_device_path = tpu_device_path
        self.enable_opencv_classification = enable_opencv_classification
        self.min_ball_size = min_ball_size

        # Model state
        self.model: Optional[Any] = None
        self.model_format: Optional[ModelFormat] = None
        self.model_loaded = False

        # TPU-specific state
        self.tpu_available = False
        self.tpu_interpreter: Optional[Any] = None
        self.tpu_input_details: Optional[list] = None
        self.tpu_output_details: Optional[list] = None

        # Thread safety for model hot-swapping
        self._model_lock = threading.RLock()

        # OpenCV classifier for ball type refinement
        self._opencv_classifier: Optional[Any] = None
        if enable_opencv_classification:
            from .balls import BallDetector

            self._opencv_classifier = BallDetector(
                {"detection_method": "combined", "debug_mode": False}
            )

        # Class mapping
        self._init_class_mapping()

        # Statistics
        self.stats = {
            "total_inferences": 0,
            "total_detections": 0,
            "avg_inference_time": 0.0,
            "fallback_mode": False,
            "using_tpu": False,
        }

        # Detect TPU availability if device is 'tpu'
        if device == "tpu":
            self.tpu_available = self._detect_tpu()
            if self.tpu_available:
                logger.info("Coral Edge TPU detected and available")
                self.stats["using_tpu"] = True
            else:
                logger.warning("TPU requested but not available, falling back to CPU")
                self.device = "cpu"

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

    def _detect_tpu(self) -> bool:
        """Detect if Coral Edge TPU is available.

        Returns:
            True if TPU is detected and pycoral is available
        """
        try:
            # Try to import pycoral
            from pycoral.utils import edgetpu

            # List available TPU devices
            devices = edgetpu.list_edge_tpus()

            if not devices:
                logger.info("No Coral Edge TPU devices found")
                return False

            # Log detected devices
            logger.info(f"Found {len(devices)} Coral Edge TPU device(s):")
            for i, device in enumerate(devices):
                device_type = device.get("type", "unknown")
                device_path = device.get("path", "unknown")
                logger.info(f"  Device {i}: type={device_type}, path={device_path}")

            return True

        except ImportError:
            logger.warning(
                "pycoral library not installed. TPU support requires pycoral. "
                "Install with: pip install pycoral"
            )
            return False
        except Exception as e:
            logger.warning(f"Error detecting TPU: {e}")
            return False

    def _load_model(self, model_path: str) -> None:
        """Load YOLO model from file.

        Args:
            model_path: Path to model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            ImportError: If ultralytics or pycoral package not installed
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
        elif suffix == ".tflite":
            self.model_format = ModelFormat.EDGETPU
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

        # Handle Edge TPU model loading separately
        if self.model_format == ModelFormat.EDGETPU:
            self._load_tpu_model(str(path))
        else:
            # Standard YOLO model loading (PyTorch, ONNX, TensorRT)
            try:
                # Import ultralytics YOLO
                from ultralytics import YOLO

                logger.info(
                    f"Loading YOLO model from {model_path} (format: {self.model_format.value})"
                )

                # Load model
                self.model = YOLO(str(path))

                # Extract class names from model
                if hasattr(self.model, "names") and self.model.names:
                    self.class_names = self.model.names
                    logger.info(
                        f"Loaded {len(self.class_names)} classes from model: {self.class_names}"
                    )
                    # Update reverse mapping
                    self.name_to_class = {v: k for k, v in self.class_names.items()}
                else:
                    logger.warning(
                        "Could not extract class names from model, using defaults"
                    )

                # Set device
                if self.device != "cpu" and self.device != "tpu":
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

    def _load_tpu_model(self, model_path: str) -> None:
        """Load Edge TPU TensorFlow Lite model.

        Args:
            model_path: Path to .tflite model file

        Raises:
            ImportError: If pycoral not installed
            RuntimeError: If model loading fails
        """
        try:
            from pycoral.utils.edgetpu import make_interpreter

            logger.info(f"Loading Edge TPU model from {model_path}")

            # Determine device specification
            device_spec = None
            if self.tpu_device_path:
                if self.tpu_device_path == "usb":
                    device_spec = "usb"
                elif self.tpu_device_path == "pcie":
                    device_spec = "pci"
                elif self.tpu_device_path.startswith("/dev/"):
                    # Specific device path
                    device_spec = self.tpu_device_path
                else:
                    logger.warning(
                        f"Unknown TPU device path format: {self.tpu_device_path}, using auto-detect"
                    )

            # Create interpreter with Edge TPU delegate
            if device_spec:
                logger.info(f"Creating TPU interpreter with device: {device_spec}")
                self.tpu_interpreter = make_interpreter(model_path, device=device_spec)
            else:
                logger.info("Creating TPU interpreter with auto-detect")
                self.tpu_interpreter = make_interpreter(model_path)

            # Allocate tensors
            self.tpu_interpreter.allocate_tensors()

            # Get input and output details
            self.tpu_input_details = self.tpu_interpreter.get_input_details()
            self.tpu_output_details = self.tpu_interpreter.get_output_details()

            # Log model details
            logger.info(f"TPU Model input shape: {self.tpu_input_details[0]['shape']}")
            logger.info(f"TPU Model input dtype: {self.tpu_input_details[0]['dtype']}")
            logger.info(f"TPU Model has {len(self.tpu_output_details)} outputs")

            self.model_loaded = True
            logger.info("Edge TPU model loaded successfully")

        except ImportError as e:
            raise ImportError(
                "pycoral library not installed. TPU support requires pycoral. "
                "Install with: pip install pycoral"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Edge TPU model: {e}") from e

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
            List of ball detections (class IDs 0-15 for multi-class models, or class 0 for single-class models)
        """
        if not self.model_loaded:
            logger.debug("Model not loaded, returning empty ball detections")
            return []

        try:
            # Run inference
            detections = self._run_inference(frame)

            # Filter for ball classes
            # Check if we have a simplified 2-class model (ball=0, cue=1) or full multi-class model (0-18)
            if (
                "ball" in self.class_names.values()
                and "cue" in self.class_names.values()
            ):
                # Simplified model: class 0 is "ball", class 1 is "cue"
                ball_detections = [
                    det
                    for det in detections
                    if det.class_name == "ball" or (0 <= det.class_id <= 15)
                ]
            else:
                # Full model: classes 0-15 are individual ball types
                ball_detections = [det for det in detections if 0 <= det.class_id <= 15]

            # Explicitly filter out cue stick detections to prevent them from being returned as balls
            # This prevents trajectory lines from being drawn from cue stick positions
            ball_detections = [
                det
                for det in ball_detections
                if det.class_name.lower() not in ["cue", "cue_stick"]
            ]

            # Filter out small detections (marker dots, noise) based on size
            # Billiard balls should have a minimum size in pixels
            filtered_detections = []
            for det in ball_detections:
                # Check if detection meets minimum size requirement
                if det.width >= self.min_ball_size and det.height >= self.min_ball_size:
                    filtered_detections.append(det)
                else:
                    logger.debug(
                        f"Filtered out small detection: {det.class_name} "
                        f"size={det.width:.1f}x{det.height:.1f}px (min={self.min_ball_size}px)"
                    )

            return filtered_detections

        except Exception as e:
            logger.error(f"Ball detection failed: {e}")
            return []

    def detect_balls_with_classification(
        self, frame: NDArray[np.uint8], min_confidence: float = 0.25
    ) -> list[Any]:
        """Detect balls and optionally classify with OpenCV refinement.

        This is the PRIMARY and RECOMMENDED method for ball detection. It provides hybrid
        YOLO+OpenCV detection that combines the strengths of both approaches:
        - YOLO provides accurate ball position and size, even in challenging lighting
        - OpenCV classifier refines ball type/number when enable_opencv_classification=True

        This method works best with the default detector settings:
        - confidence=0.15: Low threshold captures balls in difficult angles/lighting
        - enable_opencv_classification=True: Accurate ball number detection
        - min_ball_size=20: Filters out markers and noise

        Args:
            frame: Input image in BGR format
            min_confidence: Minimum confidence threshold for detections (default 0.25)

        Returns:
            List of Ball objects with position and type information

        Note:
            Prefer this method over detect_balls() when you need ball type/number information.
            The hybrid approach provides better accuracy than YOLO-only or OpenCV-only detection.
        """
        from ..models import Ball
        from .detector_adapter import yolo_detections_to_balls

        # Get YOLO detections
        yolo_detections = self.detect_balls(frame)

        if not yolo_detections:
            return []

        # Convert to Ball objects
        detected_balls = []
        for det in yolo_detections:
            # Convert Detection to dict format for adapter
            detection_dict = {
                "bbox": det.bbox,
                "confidence": det.confidence,
                "class_id": det.class_id,
                "class_name": det.class_name,
            }

            balls = yolo_detections_to_balls(
                [detection_dict],
                (frame.shape[0], frame.shape[1]),
                min_confidence=min_confidence,
                bbox_format="xyxy",
            )

            # Optionally refine with OpenCV classification
            for ball in balls:
                if self._opencv_classifier is not None and det.class_name == "ball":
                    # Extract ball region for classification
                    x, y = ball.position
                    r = ball.radius
                    x1, y1 = max(0, int(x - r * 1.2)), max(0, int(y - r * 1.2))
                    x2, y2 = min(frame.shape[1], int(x + r * 1.2)), min(
                        frame.shape[0], int(y + r * 1.2)
                    )
                    ball_region = frame[y1:y2, x1:x2]

                    if ball_region.size > 0:
                        # Classify ball type using OpenCV
                        ball_type, conf, ball_number = (
                            self._opencv_classifier.classify_ball_type(
                                ball_region, ball.position, r
                            )
                        )
                        # Update ball with classified type
                        ball.ball_type = ball_type
                        ball.number = ball_number

                detected_balls.append(ball)

        return detected_balls

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

            # Filter for cue stick class
            # Check if we have a simplified 2-class model (ball=0, cue=1) or full multi-class model
            if "cue" in self.class_names.values():
                # Simplified model or full model with "cue" class name
                cue_detections = [
                    det
                    for det in detections
                    if det.class_name == "cue"
                    or det.class_name == "cue_stick"
                    or det.class_id == 16
                ]
            else:
                # Full model: class 16 is cue_stick
                cue_detections = [det for det in detections if det.class_id == 16]

            # Return highest confidence cue detection
            if cue_detections:
                cue_detections.sort(key=lambda d: d.confidence, reverse=True)
                cue = cue_detections[0]

                # Calculate accurate angle from cue region using edge detection
                cue.angle = self._estimate_cue_angle(frame, cue)

                return cue

            return None

        except Exception as e:
            logger.error(f"Cue detection failed: {e}")
            return None

    def _estimate_cue_angle(
        self, frame: NDArray[np.uint8], cue_detection: Detection
    ) -> float:
        """Estimate cue stick angle from bounding box region using edge detection.

        Uses Hough Line Transform to find the dominant line in the cue region.

        Args:
            frame: Input image in BGR format
            cue_detection: Cue detection with bounding box

        Returns:
            Angle in degrees (0-360), where:
            - 0° = pointing right (East)
            - 90° = pointing down (South)
            - 180° = pointing left (West)
            - 270° = pointing up (North)
        """
        try:
            # Extract cue region from frame
            x1, y1, x2, y2 = cue_detection.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Add some padding to ensure we capture the full cue
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)

            cue_region = frame[y1:y2, x1:x2]

            if cue_region.size == 0:
                logger.warning("Empty cue region, using fallback angle")
                return 0.0

            # Convert to grayscale
            gray = cv2.cvtColor(cue_region, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Edge detection
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

            # Hough Line Transform to find lines
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=30,
                minLineLength=int(min(cue_region.shape[0], cue_region.shape[1]) * 0.3),
                maxLineGap=10,
            )

            if lines is None or len(lines) == 0:
                # Fallback: use bounding box aspect ratio
                logger.debug("No lines found, using bbox aspect ratio")
                if cue_detection.width > cue_detection.height:
                    return 0.0  # Horizontal
                else:
                    return 90.0  # Vertical

            # Find the longest line (most likely the cue stick)
            longest_line = None
            max_length = 0

            for line in lines:
                x1_l, y1_l, x2_l, y2_l = line[0]
                length = np.sqrt((x2_l - x1_l) ** 2 + (y2_l - y1_l) ** 2)
                if length > max_length:
                    max_length = length
                    longest_line = (x1_l, y1_l, x2_l, y2_l)

            if longest_line is None:
                logger.debug("No valid line found, using fallback")
                return 0.0

            # Calculate angle from the longest line
            x1_l, y1_l, x2_l, y2_l = longest_line

            # Calculate angle in radians, then convert to degrees
            # NOTE: This angle represents the direction from (x1_l, y1_l) to (x2_l, y2_l)
            # but Hough can return endpoints in either order, so we need to normalize later
            angle_rad = np.arctan2(y2_l - y1_l, x2_l - x1_l)
            angle_deg = np.degrees(angle_rad)

            # Normalize to 0-360 range
            if angle_deg < 0:
                angle_deg += 360

            # Store the line endpoints in the cue_detection object for orientation correction later
            # Convert from cue_region coordinates to full frame coordinates
            line_end1_x = x1_l + x1  # x1 is the region offset
            line_end1_y = y1_l + y1  # y1 is the region offset
            line_end2_x = x2_l + x1
            line_end2_y = y2_l + y1
            cue_detection.line_end1 = (line_end1_x, line_end1_y)
            cue_detection.line_end2 = (line_end2_x, line_end2_y)

            # Store the line center in the cue_detection object for positioning
            line_center_x = (line_end1_x + line_end2_x) / 2
            line_center_y = (line_end1_y + line_end2_y) / 2
            cue_detection.line_center = (line_center_x, line_center_y)

            logger.debug(
                f"Estimated cue angle: {angle_deg:.1f}° (line: {longest_line}, length: {max_length:.1f}, center: ({line_center_x:.1f}, {line_center_y:.1f}))"
            )

            return angle_deg

        except Exception as e:
            logger.error(f"Angle estimation failed: {e}", exc_info=True)
            # Fallback to simple heuristic
            if cue_detection.width > cue_detection.height:
                return 0.0
            else:
                return 90.0

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

        # Route to appropriate inference method based on model format
        if self.model_format == ModelFormat.EDGETPU:
            return self._run_tpu_inference(frame)
        else:
            return self._run_standard_inference(frame)

    def _run_standard_inference(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run standard YOLO inference (PyTorch, ONNX, TensorRT).

        Args:
            frame: Input image in BGR format

        Returns:
            List of all detections
        """
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

    def _run_tpu_inference(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run Edge TPU inference on frame.

        Args:
            frame: Input image in BGR format

        Returns:
            List of all detections
        """
        import time

        start_time = time.time()

        try:
            # Acquire lock for thread-safe TPU access
            with self._model_lock:
                if (
                    not self.model_loaded
                    or self.tpu_interpreter is None
                    or self.tpu_input_details is None
                ):
                    return []

                # Get input shape from model
                input_shape = self.tpu_input_details[0]["shape"]
                input_height, input_width = input_shape[1], input_shape[2]

                # Preprocess frame for TPU
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize to model input size
                frame_resized = cv2.resize(
                    frame_rgb,
                    (input_width, input_height),
                    interpolation=cv2.INTER_LINEAR,
                )

                # Normalize to uint8 (TPU models typically use quantized uint8 input)
                # Check input dtype
                input_dtype = self.tpu_input_details[0]["dtype"]
                if input_dtype == np.uint8:
                    input_tensor = frame_resized
                else:
                    # Normalize to float32 [0, 1]
                    input_tensor = frame_resized.astype(np.float32) / 255.0

                # Add batch dimension
                input_tensor = np.expand_dims(input_tensor, axis=0)

                # Set input tensor
                self.tpu_interpreter.set_tensor(
                    self.tpu_input_details[0]["index"], input_tensor
                )

                # Run inference
                self.tpu_interpreter.invoke()

                # Get output tensors
                # YOLOv8 Edge TPU models typically have multiple outputs:
                # - Bounding boxes (x, y, w, h)
                # - Confidence scores
                # - Class IDs
                outputs = []
                for output_detail in self.tpu_output_details:
                    output = self.tpu_interpreter.get_tensor(output_detail["index"])
                    outputs.append(output)

            # Parse TPU output to Detection objects
            detections = self._parse_tpu_output(
                outputs, frame.shape, (input_height, input_width)
            )

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
                f"TPU inference: {len(detections)} detections in {inference_time:.1f}ms"
            )

            return detections

        except Exception as e:
            logger.error(f"TPU inference failed: {e}")
            return []

    def _parse_tpu_output(
        self,
        outputs: list[NDArray],
        original_shape: tuple[int, int, int],
        model_input_shape: tuple[int, int],
    ) -> list[Detection]:
        """Parse Edge TPU model output to Detection objects.

        Args:
            outputs: List of output tensors from TPU model
            original_shape: Original frame shape (H, W, C)
            model_input_shape: Model input shape (H, W)

        Returns:
            List of Detection objects
        """
        detections = []

        # YOLOv8 Edge TPU output format typically:
        # Output 0: [1, num_predictions, 84] where 84 = 4 (bbox) + 80 (classes)
        # or [1, 84, num_predictions] depending on model export
        if len(outputs) == 0:
            return detections

        # Get main output tensor
        output = outputs[0]

        # Handle different output formats
        if len(output.shape) == 3:
            # Format: [1, num_predictions, 84] or [1, 84, num_predictions]
            if output.shape[-1] > output.shape[1]:
                # [1, num_predictions, features]
                predictions = output[0]  # Remove batch dimension
            else:
                # [1, features, num_predictions] - transpose
                predictions = output[0].T

            # Parse predictions
            num_predictions = predictions.shape[0]
            predictions.shape[1] - 4  # Subtract bbox coordinates

            # Calculate scale factors for coordinate conversion
            orig_h, orig_w = original_shape[:2]
            model_h, model_w = model_input_shape
            scale_x = orig_w / model_w
            scale_y = orig_h / model_h

            for i in range(num_predictions):
                pred = predictions[i]

                # Extract bbox (cx, cy, w, h format in YOLO)
                cx, cy, w, h = pred[:4]

                # Extract class scores
                class_scores = pred[4:]

                # Get class with highest confidence
                class_id = int(np.argmax(class_scores))
                confidence = float(class_scores[class_id])

                # Apply confidence threshold
                if confidence < self.confidence:
                    continue

                # Convert to xyxy format and scale to original image size
                x1 = (cx - w / 2) * scale_x
                y1 = (cy - h / 2) * scale_y
                x2 = (cx + w / 2) * scale_x
                y2 = (cy + h / 2) * scale_y

                # Calculate center and dimensions in original image
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                # Get class name
                class_name = self.class_names.get(class_id, f"unknown_{class_id}")

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

        # Apply NMS to remove overlapping detections
        if detections:
            detections = self._apply_nms(detections, self.nms_threshold)

        return detections

    def _apply_nms(
        self, detections: list[Detection], iou_threshold: float
    ) -> list[Detection]:
        """Apply Non-Maximum Suppression to remove overlapping detections.

        Args:
            detections: List of Detection objects
            iou_threshold: IoU threshold for NMS

        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []

        # Sort detections by confidence (descending)
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        # Group detections by class
        class_groups = {}
        for det in detections:
            if det.class_id not in class_groups:
                class_groups[det.class_id] = []
            class_groups[det.class_id].append(det)

        # Apply NMS per class
        final_detections = []
        for class_id, class_dets in class_groups.items():
            # Convert to numpy arrays for IoU calculation
            boxes = np.array([d.bbox for d in class_dets])
            scores = np.array([d.confidence for d in class_dets])

            # OpenCV NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), scores.tolist(), self.confidence, iou_threshold
            )

            # Extract kept detections
            if len(indices) > 0:
                indices = indices.flatten()
                for idx in indices:
                    final_detections.append(class_dets[idx])

        return final_detections

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
        info = {
            "model_path": str(self.model_path) if self.model_path else None,
            "model_format": self.model_format.value if self.model_format else None,
            "model_loaded": self.model_loaded,
            "device": self.device,
            "confidence_threshold": self.confidence,
            "nms_threshold": self.nms_threshold,
            "fallback_mode": self.stats["fallback_mode"],
            "using_tpu": self.stats.get("using_tpu", False),
        }

        # Add TPU-specific info if using TPU
        if self.model_format == ModelFormat.EDGETPU and self.tpu_interpreter:
            info["tpu_available"] = self.tpu_available
            info["tpu_device_path"] = self.tpu_device_path
            if self.tpu_input_details:
                info["tpu_input_shape"] = self.tpu_input_details[0]["shape"].tolist()
                info["tpu_input_dtype"] = str(self.tpu_input_details[0]["dtype"])

        return info

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
            - device: 'cpu', 'cuda', 'mps', 'tpu'
            - confidence: float 0.0-1.0 (default 0.15 for optimal ball detection)
            - nms_threshold: float 0.0-1.0
            - auto_fallback: bool
            - enable_opencv_classification: bool (default True for hybrid YOLO+OpenCV detection)
            - tpu_device_path: str (optional, for TPU)
            - min_ball_size: int (optional, minimum ball size in pixels, default 20)

    Returns:
        Configured YOLODetector instance

    Note:
        Uses working defaults from VisionModule: confidence=0.15, enable_opencv_classification=True.
        These values are proven to work well for billiards ball detection.
    """
    if config is None:
        config = {}

    return YOLODetector(
        model_path=model_path,
        device=config.get("device", "cpu"),
        confidence=config.get("confidence", 0.15),
        nms_threshold=config.get("nms_threshold", 0.45),
        auto_fallback=config.get("auto_fallback", True),
        enable_opencv_classification=config.get("enable_opencv_classification", True),
        tpu_device_path=config.get("tpu_device_path"),
        min_ball_size=config.get("min_ball_size", 20),
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
