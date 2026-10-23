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
from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Hashable,
    Optional,
    TypeVar,
    cast,
)

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from backend.common.models import Ball

_OutcomeT = TypeVar("_OutcomeT")


class ModelValidationError(Exception):
    """Raised when model validation fails."""

    pass


class ModelFormat(Enum):
    """Supported model formats."""

    PYTORCH = "pt"
    ONNX = "onnx"
    TENSORRT = "engine"
    EDGETPU = "tflite"  # Edge TPU TensorFlow Lite
    COREML = "mlpackage"  # Native CoreML model


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


class BallDetectionStatus(str, Enum):
    """Model-independent result status for a ball observation attempt."""

    DETECTIONS = "detections"
    NO_DETECTIONS = "no_detections"
    MODEL_UNAVAILABLE = "model_unavailable"
    FAILED = "failed"


class DetectionFailureStage(str, Enum):
    """Safe, bounded categories for detector failure reporting."""

    INFERENCE = "inference"
    DECODING = "decoding"
    CLASSIFICATION = "classification"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BallDetectionOutcome(Generic[_OutcomeT]):
    """Status-bearing result of attempting to observe balls in one frame.

    A successful empty list is represented by ``NO_DETECTIONS``. It is distinct
    from a missing model and from an inference or decoding failure, which is
    essential when consumers assess observation freshness and detector health.
    Failure details intentionally contain only a bounded stage category; raw
    backend exception text can include paths, frame data, or runtime internals.
    """

    status: BallDetectionStatus
    detections: tuple[_OutcomeT, ...] = ()
    failure_stage: DetectionFailureStage | None = None

    def __post_init__(self) -> None:
        has_detections = bool(self.detections)
        if (self.status == BallDetectionStatus.DETECTIONS) != has_detections:
            raise ValueError("detection outcome status does not match its detections")
        if (self.status == BallDetectionStatus.FAILED) != (
            self.failure_stage is not None
        ):
            raise ValueError("only failed outcomes may include a failure stage")


class _ModelUnavailableError(RuntimeError):
    """Internal signal that no usable inference backend is currently loaded."""


class _DetectionBackendFailure(RuntimeError):
    """Internal failure carrying only a safe inference/decoding stage."""

    def __init__(self, stage: DetectionFailureStage) -> None:
        super().__init__(stage.value)
        self.stage = stage


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

    # Additional cue stick line detection fields (populated by _estimate_cue_angle)
    line_end1: Optional[tuple[float, float]] = None  # Line endpoint 1 (x, y)
    line_end2: Optional[tuple[float, float]] = None  # Line endpoint 2 (x, y)
    line_center: Optional[tuple[float, float]] = None  # Line center (x, y)


@dataclass
class TableElements:
    """Detected table elements from YOLO.

    Represents table surface, pockets, and other structural elements.
    """

    table_bbox: Optional[tuple[float, float, float, float]] = None
    table_confidence: float = 0.0
    pockets: list[Detection] = field(default_factory=list)
    rails: list[Detection] = field(default_factory=list)


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
        enable_opencv_classification: bool = False,
        min_ball_size: int = 20,
        camera_resolution: tuple[int, int] = (1920, 1080),
    ) -> None:
        """Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model file (.pt, .onnx, or .tflite). If None, detector
                       will operate in fallback mode without YOLO.
            device: Device to run inference on ('cpu', 'cuda', 'mps', 'coreml', 'tpu')
            confidence: Minimum confidence threshold (0.0-1.0). Default 0.15 works well for
                       billiards ball detection while capturing difficult angles and lighting.
            nms_threshold: Non-maximum suppression IoU threshold
            auto_fallback: Automatically fall back to None model if loading fails
            tpu_device_path: Coral TPU device path (e.g., '/dev/bus/usb/001/002', 'usb', 'pcie', or None for auto)
            enable_opencv_classification: Enable the experimental OpenCV color/stripe
                heuristic. It can guess ball numbers, but the configured YOLO model
                only detects generic balls and does not validate those guesses.
            min_ball_size: Minimum ball size in pixels (width or height) to filter out small detections
                          like markers and noise. Default 20px works well for typical camera setups.
            camera_resolution: Camera resolution (width, height) for scaling pixel-based parameters in
                             OpenCV classifier. Defaults to 1080p.

        Raises:
            FileNotFoundError: If model_path is provided but file doesn't exist
            RuntimeError: If model loading fails and auto_fallback is False

        Note:
            These defaults are retained for the legacy Python VisionModule path.
            Historical Python debugger results do not qualify the Rust detector's
            accuracy, recall, or false-positive rate.
        """
        self.model_path = model_path
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.auto_fallback = auto_fallback
        self.tpu_device_path = tpu_device_path
        self.enable_opencv_classification = enable_opencv_classification
        self.min_ball_size = min_ball_size
        self.camera_resolution = camera_resolution
        # Set by VisionModule only during an explicitly enabled audit session.
        self.audit_profiler: Any | None = None

        # Auto-detect best device if not specified or if "auto"
        if device == "auto" or device is None:
            self.device = self._auto_detect_device()
            logger.info(f"Auto-detected device: {self.device}")
        else:
            self.device = device

        # Model state
        self.model: Optional[Any] = None
        self.model_format: Optional[ModelFormat] = None
        self.model_loaded = False

        # ONNX Runtime session for direct CoreML inference
        self.onnx_session: Optional[Any] = None
        self.onnx_input_name: Optional[str] = None
        self.onnx_output_names: Optional[list[str]] = None

        # TPU-specific state
        self.tpu_available = False
        self.tpu_interpreter: Optional[Any] = None
        self.tpu_input_details: Optional[list[Any]] = None
        self.tpu_output_details: Optional[list[Any]] = None

        # Thread safety for model hot-swapping
        self._model_lock = threading.RLock()

        # Cache only a caller-identified frame/view. Buffer identity and sampled
        # pixels are not reliable frame identities when camera memory is reused.
        self._detection_cache: dict[Hashable, list[Detection]] = {}
        self._cache_lock = threading.Lock()
        self._inference_model_generation = 0

        # OpenCV classifier for ball type refinement
        self._opencv_classifier: Optional[Any] = None
        if enable_opencv_classification:
            from .balls import BallDetector

            self._opencv_classifier = BallDetector(
                {"detection_method": "combined", "debug_mode": False},
                camera_resolution=camera_resolution,
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

    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device for inference.

        Priority:
        1. Native CoreML for a CoreML package on macOS
        2. CUDA for formats supported by the PyTorch inference path
        3. CoreML for ONNX on Apple Silicon when the provider is available
        4. MPS for PyTorch weights on Apple Silicon
        5. CPU (fallback)

        Returns:
            Device string: 'cuda', 'coreml', 'mps', or 'cpu'
        """
        import platform

        # A .mlpackage selects Ultralytics' native CoreML backend from its
        # format. Reporting MPS or CUDA for this artifact is misleading because
        # those device arguments are not used by the CoreML predictor.
        if (
            self.model_path
            and Path(self.model_path).suffix.lower() == ".mlpackage"
            and platform.system() == "Darwin"
        ):
            logger.info("CoreML package detected on macOS - using native CoreML")
            return "coreml"

        # Check for CUDA GPU
        try:
            import torch

            if torch.cuda.is_available():
                logger.info("CUDA GPU detected")
                return "cuda"
        except ImportError:
            pass

        # Check for Apple Silicon and determine best option
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            # Check if we'll be using ONNX model (need to know model format)
            if self.model_path and self.model_path.endswith(".onnx"):
                # For ONNX on Apple Silicon, use CoreML execution provider
                try:
                    ort = cast("Any", import_module("onnxruntime"))

                    if "CoreMLExecutionProvider" in ort.get_available_providers():
                        logger.info(
                            "Apple Silicon detected with ONNX model - using CoreML"
                        )
                        return "coreml"
                except ImportError:
                    pass

            # For PyTorch models on Apple Silicon, use MPS
            try:
                import torch

                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    logger.info("Apple Silicon detected with PyTorch - using MPS")
                    return "mps"
            except ImportError:
                pass

        # Fallback to CPU
        logger.info("No GPU acceleration detected - using CPU")
        return "cpu"

    def _detect_tpu(self) -> bool:
        """Detect if Coral Edge TPU is available.

        Returns:
            True if TPU is detected and pycoral is available
        """
        try:
            # Try to import pycoral
            edgetpu = import_module("pycoral.utils.edgetpu")
            list_edge_tpus = cast(
                "Callable[[], list[dict[str, str]]]",
                edgetpu.__dict__["list_edge_tpus"],
            )

            # List available TPU devices.
            devices = list_edge_tpus()

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

        # Determine model format from extension or name
        # Check for .mlpackage directory (CoreML model)
        if path.suffix == ".mlpackage" or path.name.endswith(".mlpackage"):
            self.model_format = ModelFormat.COREML
        else:
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
        elif self.model_format == ModelFormat.COREML:
            # CoreML models are loaded directly by Ultralytics
            self._load_coreml_model(str(path))
        else:
            # Standard YOLO model loading (PyTorch, ONNX, TensorRT)
            try:
                # Import ultralytics YOLO
                import os

                ultralytics = import_module("ultralytics")
                # Accessing the lazy module attribute triggers Ultralytics loading.
                YOLO = cast("Callable[..., Any]", ultralytics.YOLO)

                logger.info(
                    f"Loading YOLO model from {model_path} (format: {self.model_format.value})"
                )

                # Configure CoreML execution provider for ONNX models on Apple Silicon
                if self.device == "coreml" and self.model_format == ModelFormat.ONNX:
                    logger.info(
                        "Configuring ONNX Runtime to use CoreML execution provider"
                    )

                    # Import onnxruntime to use CoreML directly
                    try:
                        ort = cast("Any", import_module("onnxruntime"))

                        available_providers = ort.get_available_providers()
                        if "CoreMLExecutionProvider" in available_providers:
                            logger.info(
                                f"CoreML execution provider is available. All providers: {available_providers}"
                            )

                            # Use ONNX Runtime directly with CoreML provider
                            # This bypasses Ultralytics which doesn't support CoreML with ONNX
                            logger.info(
                                "Creating ONNX Runtime session with CoreML provider"
                            )

                            # Configure session options
                            sess_options = ort.SessionOptions()
                            sess_options.graph_optimization_level = (
                                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                            )

                            # Create session with CoreML provider (highest priority)
                            self.onnx_session = ort.InferenceSession(
                                str(path),
                                sess_options=sess_options,
                                providers=[
                                    "CoreMLExecutionProvider",
                                    "CPUExecutionProvider",
                                ],
                            )

                            # Get input/output names for inference
                            self.onnx_input_name = self.onnx_session.get_inputs()[
                                0
                            ].name
                            self.onnx_output_names = [
                                output.name
                                for output in self.onnx_session.get_outputs()
                            ]

                            # Log which provider is actually being used
                            active_providers = self.onnx_session.get_providers()
                            logger.info(
                                f"ONNX Runtime session created. Active providers: {active_providers}"
                            )

                            if "CoreMLExecutionProvider" in active_providers:
                                logger.info(
                                    "✓ Successfully using CoreML execution provider"
                                )
                            else:
                                logger.warning(
                                    f"CoreML provider not active, using: {active_providers[0]}"
                                )

                            # Load model with Ultralytics just to get class names
                            # We won't use it for inference
                            logger.info(
                                "Loading model metadata with Ultralytics (for class names only)"
                            )
                            self.model = YOLO(str(path))

                        else:
                            logger.warning(
                                "CoreML provider not available, falling back to CPU with Ultralytics"
                            )
                            self.device = "cpu"
                            # Load model normally
                            self.model = YOLO(str(path))
                    except ImportError:
                        logger.warning(
                            "onnxruntime not found, falling back to Ultralytics"
                        )
                        self.device = "cpu"
                        # Load model normally
                        self.model = YOLO(str(path))
                else:
                    # Load model normally for non-CoreML cases
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

                self.model_loaded = True
                logger.info(
                    "YOLO model loaded; requested inference device: %s",
                    self.device,
                )

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
            edgetpu = import_module("pycoral.utils.edgetpu")
            make_interpreter = cast(
                "Callable[..., Any]", edgetpu.__dict__["make_interpreter"]
            )

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

    def _load_coreml_model(self, model_path: str) -> None:
        """Load native CoreML model (.mlpackage).

        Args:
            model_path: Path to .mlpackage directory

        Raises:
            ImportError: If Ultralytics or a required model runtime dependency is
                unavailable
            RuntimeError: If model loading fails
        """
        try:
            ultralytics = import_module("ultralytics")
        except ImportError as e:
            raise ImportError(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            ) from e

        try:
            # Ultralytics exposes YOLO lazily, so it may not exist in the
            # module dictionary until attribute access resolves the export.
            YOLO = cast("Callable[..., Any]", ultralytics.YOLO)

            logger.info(f"Loading CoreML model from {model_path}")

            # Ultralytics directly supports .mlpackage files on macOS
            # Load the model - it will use CoreML natively
            self.model = YOLO(str(model_path))

            # Extract class names from model
            if hasattr(self.model, "names") and self.model.names:
                self.class_names = self.model.names
                logger.info(
                    f"Loaded {len(self.class_names)} classes from CoreML model: {self.class_names}"
                )
                # Update reverse mapping
                self.name_to_class = {v: k for k, v in self.class_names.items()}
            else:
                logger.warning(
                    "Could not extract class names from CoreML model, using defaults"
                )

            self.model_loaded = True
            logger.info(
                "✓ CoreML model loaded successfully - using native CoreML acceleration"
            )

        except ImportError:
            # Preserve missing runtime dependencies (for example coremltools)
            # raised while resolving YOLO or constructing the native model.
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load CoreML model: {e}") from e

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

            metadata: dict[str, Any] = {
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
            start_time = time.perf_counter()
            detections = detector._run_inference(test_frame)
            inference_time = (time.perf_counter() - start_time) * 1000

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

    def set_background_frame(self, frame: NDArray[np.uint8]) -> None:
        """Set the empty-table reference used by OpenCV classification.

        YOLO itself does not subtract a background. When its optional OpenCV
        classifier is enabled, forward the reference frame to that detector so
        its motion mask and table geometry remain available to classification.
        """
        classifier = self._opencv_classifier
        if classifier is not None:
            classifier.set_background_frame(frame)

    def detect_balls(
        self, frame: NDArray[np.uint8], *, frame_key: Optional[Hashable] = None
    ) -> list[Detection]:
        """Compatibility API returning detections or an empty list.

        Args:
            frame: Input image in BGR format
            frame_key: Optional identity for this source-frame/view. Reuse the
                same key for other detection methods called on the same pixels.
                A changed frame or crop must use a different key. Calls without
                a key always run inference and cannot return stale cached data.

        Returns:
            List of ball detections (class IDs 0-15 for multi-class models, or class 0 for single-class models)
        """
        return list(self.detect_balls_outcome(frame, frame_key=frame_key).detections)

    def detect_balls_outcome(
        self, frame: NDArray[np.uint8], *, frame_key: Optional[Hashable] = None
    ) -> BallDetectionOutcome[Detection]:
        """Detect balls and preserve whether the frame was actually observed.

        ``NO_DETECTIONS`` means a loaded model completed inference and decoding
        successfully but found no qualifying balls. ``MODEL_UNAVAILABLE`` means
        no usable backend was loaded. ``FAILED`` means inference or decoding did
        not complete; ``failure_stage`` reports only a bounded stage category,
        never raw exception text from the model runtime.

        This API is independent of backend format. The legacy ``detect_balls``
        method remains a list-returning wrapper and consequently maps the last
        three statuses to an empty list for compatibility.
        """
        if not getattr(self, "model_loaded", False):
            logger.debug("Model not loaded; ball observation unavailable")
            return BallDetectionOutcome(BallDetectionStatus.MODEL_UNAVAILABLE)

        try:
            detections = self._run_inference(frame, frame_key=frame_key)
            ball_detections = self._filter_ball_detections(detections)
        except _ModelUnavailableError:
            return BallDetectionOutcome(BallDetectionStatus.MODEL_UNAVAILABLE)
        except _DetectionBackendFailure as error:
            logger.error("Ball detection failed during %s", error.stage.value)
            return BallDetectionOutcome(
                BallDetectionStatus.FAILED, failure_stage=error.stage
            )
        except Exception as error:
            # Keep unexpected cache/integration failures observable without
            # leaking arbitrary exception text or runtime-specific details.
            logger.error(
                "Ball detection failed in an unexpected stage (%s)",
                type(error).__name__,
            )
            return BallDetectionOutcome(
                BallDetectionStatus.FAILED,
                failure_stage=DetectionFailureStage.UNKNOWN,
            )

        if ball_detections:
            return BallDetectionOutcome(
                BallDetectionStatus.DETECTIONS, tuple(ball_detections)
            )
        return BallDetectionOutcome(BallDetectionStatus.NO_DETECTIONS)

    def _filter_ball_detections(self, detections: list[Detection]) -> list[Detection]:
        """Apply existing class, cue, and minimum-size filters consistently."""
        # Check if this is a simplified 2-class model (ball=0, cue=1) or a
        # full multi-class model (classes 0-18).
        if "ball" in self.class_names.values() and "cue" in self.class_names.values():
            ball_detections = [
                det
                for det in detections
                if det.class_name == "ball" or (0 <= det.class_id <= 15)
            ]
        else:
            ball_detections = [det for det in detections if 0 <= det.class_id <= 15]

        # Explicitly remove cue detections, then filter marker-sized boxes.
        ball_detections = [
            det
            for det in ball_detections
            if det.class_name.lower() not in ["cue", "cue_stick"]
        ]
        filtered_detections: list[Detection] = []
        for detection in ball_detections:
            max_dimension = max(detection.width, detection.height)
            min_dimension = min(detection.width, detection.height)
            aspect_ratio = max_dimension / min_dimension if min_dimension > 0 else 999
            if max_dimension >= self.min_ball_size and aspect_ratio < 3.0:
                filtered_detections.append(detection)
            else:
                logger.debug(
                    "Filtered out detection: %s size=%.1fx%.1fpx "
                    "(min=%spx, aspect=%.2f)",
                    detection.class_name,
                    detection.width,
                    detection.height,
                    self.min_ball_size,
                    aspect_ratio,
                )
        return filtered_detections

    def detect_balls_with_classification(
        self,
        frame: NDArray[np.uint8],
        min_confidence: float = 0.25,
        *,
        frame_key: Optional[Hashable] = None,
        classification_frame: Optional[NDArray[np.uint8]] = None,
    ) -> list["Ball"]:
        """Detect balls and optionally classify with OpenCV refinement.

        YOLO provides ball position and size. The optional OpenCV classifier
        guesses ball types and numbers from color and stripe patterns; it is not
        a trained identity model and should not be treated as authoritative.

        Args:
            frame: Input image in BGR format
            min_confidence: Minimum confidence threshold for detections (default 0.25)
            frame_key: Optional source-frame/view identity shared with other
                detector calls for the same image.
            classification_frame: Unmasked source image used for color
                classification. It must have the same height and width as
                ``frame`` so detector-pixel boxes remain aligned. If omitted,
                ``frame`` is used.

        Returns:
            List of Ball objects with position and type information

        Note:
            Prefer this method over detect_balls() when you need ball type/number information.
            The hybrid approach provides better accuracy than YOLO-only or OpenCV-only detection.
        """
        return list(
            self.detect_balls_with_classification_outcome(
                frame,
                min_confidence,
                frame_key=frame_key,
                classification_frame=classification_frame,
            ).detections
        )

    def detect_balls_with_classification_outcome(
        self,
        frame: NDArray[np.uint8],
        min_confidence: float = 0.25,
        *,
        frame_key: Optional[Hashable] = None,
        classification_frame: Optional[NDArray[np.uint8]] = None,
    ) -> BallDetectionOutcome["Ball"]:
        """Detect and classify balls while retaining detector and classifier status.

        Inference statuses are preserved unchanged. A successful model result is
        reported as ``DETECTIONS`` only after conversion and optional color
        classification completes and yields at least one Ball. Failures during
        adaptation or classification return ``FAILED`` with the safe
        ``CLASSIFICATION`` stage; backend exception text is not exposed.
        """
        outcome = self._get_classification_source_outcome(frame, frame_key=frame_key)
        if outcome.status != BallDetectionStatus.DETECTIONS:
            return BallDetectionOutcome(
                outcome.status, failure_stage=outcome.failure_stage
            )

        try:
            # YOLO boxes are in detector pixels. Use the unmasked source image
            # for color refinement when it is aligned with the detector input.
            classification_image = frame
            if (
                classification_frame is not None
                and classification_frame.shape[:2] == frame.shape[:2]
            ):
                classification_image = classification_frame
            elif classification_frame is not None:
                logger.warning(
                    "Ignoring classification frame with dimensions that do not "
                    "match the detector input"
                )

            crop_padding_ratio = 1.2
            try:
                from ...config import config

                crop_padding_ratio = float(
                    config.get(
                        "vision.detection.classification_crop_padding_ratio",
                        crop_padding_ratio,
                    )
                )
            except (ImportError, TypeError, ValueError):
                logger.debug("Using default ball-classification crop padding")
            crop_padding_ratio = max(crop_padding_ratio, 1.0)

            profiler = getattr(self, "audit_profiler", None)
            if profiler is None:
                balls = self._adapt_and_classify_balls(
                    list(outcome.detections),
                    frame,
                    classification_image,
                    min_confidence,
                    crop_padding_ratio,
                )
            else:
                with profiler.span("classification_and_adaptation"):
                    balls = self._adapt_and_classify_balls(
                        list(outcome.detections),
                        frame,
                        classification_image,
                        min_confidence,
                        crop_padding_ratio,
                    )
        except Exception as error:
            logger.error(
                "Ball adaptation/classification failed (%s)",
                type(error).__name__,
            )
            return BallDetectionOutcome(
                BallDetectionStatus.FAILED,
                failure_stage=DetectionFailureStage.CLASSIFICATION,
            )

        if balls:
            return BallDetectionOutcome(BallDetectionStatus.DETECTIONS, tuple(balls))
        return BallDetectionOutcome(BallDetectionStatus.NO_DETECTIONS)

    def _get_classification_source_outcome(
        self, frame: NDArray[np.uint8], *, frame_key: Optional[Hashable]
    ) -> BallDetectionOutcome[Detection]:
        """Use status-aware inference unless a legacy detector method overrides it.

        Integrations have historically replaced or subclassed ``detect_balls``
        with lightweight implementations. Preserve that extension seam by
        treating its returned list as a completed observation. Such overrides
        cannot express model-unavailable versus successful-empty, so exceptions
        map to ``FAILED/UNKNOWN`` and a returned empty list maps to
        ``NO_DETECTIONS``.
        """
        instance_override = "detect_balls" in self.__dict__
        class_override = (
            getattr(type(self), "detect_balls", None) is not YOLODetector.detect_balls
        )
        if not instance_override and not class_override:
            return self.detect_balls_outcome(frame, frame_key=frame_key)

        try:
            detections = list(self.detect_balls(frame, frame_key=frame_key))
        except Exception as error:
            logger.error(
                "Legacy ball detector override failed (%s)",
                type(error).__name__,
            )
            return BallDetectionOutcome(
                BallDetectionStatus.FAILED,
                failure_stage=DetectionFailureStage.UNKNOWN,
            )

        if detections:
            return BallDetectionOutcome(
                BallDetectionStatus.DETECTIONS, tuple(detections)
            )
        return BallDetectionOutcome(BallDetectionStatus.NO_DETECTIONS)

    def _adapt_and_classify_balls(
        self,
        yolo_detections: list[Detection],
        frame: NDArray[np.uint8],
        classification_image: NDArray[np.uint8],
        min_confidence: float,
        crop_padding_ratio: float,
    ) -> list["Ball"]:
        """Convert model boxes into balls and optionally refine their colors."""
        from .detector_adapter import yolo_detections_to_balls

        # Convert to Ball objects
        detected_balls: list[Ball] = []
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
                    # Ball.position/radius are 4K canonical; the detection box
                    # remains in this input image's pixel coordinates. Crop
                    # from the detector box to avoid mixing those spaces.
                    x1, y1, x2, y2 = det.bbox
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    radius_px = max(x2 - x1, y2 - y1) / 2.0
                    margin = radius_px * (crop_padding_ratio - 1.0)
                    crop_x1 = max(0, int(np.floor(x1 - margin)))
                    crop_y1 = max(0, int(np.floor(y1 - margin)))
                    crop_x2 = min(
                        classification_image.shape[1], int(np.ceil(x2 + margin))
                    )
                    crop_y2 = min(
                        classification_image.shape[0], int(np.ceil(y2 + margin))
                    )
                    ball_region = classification_image[crop_y1:crop_y2, crop_x1:crop_x2]

                    if ball_region.size > 0:
                        # Radius is kept in the same camera-pixel space as the crop.
                        ball_type, conf, ball_number = (
                            self._opencv_classifier.classify_ball_type(
                                ball_region, (center_x, center_y), radius_px
                            )
                        )
                        # Update ball with classified type
                        ball.ball_type = ball_type
                        ball.number = ball_number

                detected_balls.append(ball)

        return detected_balls

    def detect_cue(
        self, frame: NDArray[np.uint8], *, frame_key: Optional[Hashable] = None
    ) -> Optional[Detection]:
        """Detect cue stick in the frame.

        Args:
            frame: Input image in BGR format
            frame_key: Optional identity for this source-frame/view. Use the
                same key as ``detect_balls`` to share one inference pass.

        Returns:
            Cue stick detection or None if not detected
        """
        import time

        method_start = time.perf_counter()

        if not self.model_loaded:
            logger.debug("Model not loaded, returning no cue detection")
            return None

        try:
            # Run inference
            inference_start = time.perf_counter()
            detections = self._run_inference(frame, frame_key=frame_key)
            inference_time = (time.perf_counter() - inference_start) * 1000

            # Filter for cue stick class
            filter_start = time.perf_counter()
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
            filter_time = (time.perf_counter() - filter_start) * 1000

            # Return highest confidence cue detection
            if cue_detections:
                sort_start = time.perf_counter()
                cue_detections.sort(key=lambda d: d.confidence, reverse=True)
                cue = cue_detections[0]
                sort_time = (time.perf_counter() - sort_start) * 1000

                # Calculate accurate angle from cue region using edge detection
                angle_start = time.perf_counter()
                cue.angle = self._estimate_cue_angle(frame, cue)
                angle_time = (time.perf_counter() - angle_start) * 1000

                total_time = (time.perf_counter() - method_start) * 1000
                logger.debug(
                    f"[PERF] detect_cue timing: total={total_time:.1f}ms "
                    f"(inference={inference_time:.1f}ms, filter={filter_time:.1f}ms, "
                    f"sort={sort_time:.1f}ms, angle={angle_time:.1f}ms)"
                )

                return cue

            total_time = (time.perf_counter() - method_start) * 1000
            logger.debug(f"detect_cue timing: total={total_time:.1f}ms (no cue found)")
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
        import time

        method_start = time.perf_counter()

        try:
            # Extract cue region from frame
            extract_start = time.perf_counter()
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (
                float(coordinate) for coordinate in cue_detection.bbox
            )
            x1, y1, x2, y2 = (
                int(bbox_x1),
                int(bbox_y1),
                int(bbox_x2),
                int(bbox_y2),
            )

            # Add some padding to ensure we capture the full cue
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)

            cue_region = frame[y1:y2, x1:x2]
            extract_time = (time.perf_counter() - extract_start) * 1000

            if cue_region.size == 0:
                logger.warning("Empty cue region, using fallback angle")
                return 0.0

            # Convert to grayscale
            gray_start = time.perf_counter()
            gray = cv2.cvtColor(cue_region, cv2.COLOR_BGR2GRAY)
            gray_time = (time.perf_counter() - gray_start) * 1000

            # Apply Gaussian blur to reduce noise
            blur_start = time.perf_counter()
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            blur_time = (time.perf_counter() - blur_start) * 1000

            # Edge detection
            canny_start = time.perf_counter()
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            canny_time = (time.perf_counter() - canny_start) * 1000

            # Hough Line Transform to find lines
            hough_start = time.perf_counter()
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=30,
                minLineLength=int(min(cue_region.shape[0], cue_region.shape[1]) * 0.3),
                maxLineGap=10,
            )
            hough_time = (time.perf_counter() - hough_start) * 1000

            if lines is None or len(lines) == 0:
                # Fallback: use bounding box aspect ratio
                logger.debug("No lines found, using bbox aspect ratio")
                if cue_detection.width > cue_detection.height:
                    return 0.0  # Horizontal
                else:
                    return 90.0  # Vertical

            # A cue can be nearly parallel to a table rail. Estimate the shaft
            # axis from aligned, collinear Hough fragments instead of rejecting
            # that orientation and accidentally choosing a shorter hand edge.
            segments: list[dict[str, Any]] = []
            for line in np.asarray(lines).reshape(-1, 4):
                local_x1, local_y1, local_x2, local_y2 = (
                    float(coordinate) for coordinate in line
                )
                start = (local_x1 + x1, local_y1 + y1)
                end = (local_x2 + x1, local_y2 + y1)
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = float(np.hypot(dx, dy))
                if length == 0:
                    continue
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "midpoint": ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2),
                        "length": length,
                        "angle": float(np.degrees(np.arctan2(dy, dx)) % 180.0),
                    }
                )

            if not segments:
                logger.debug("No usable Hough segments found, using bbox aspect ratio")
                return 0.0 if cue_detection.width > cue_detection.height else 90.0

            bbox_width = max(1.0, bbox_x2 - bbox_x1)
            bbox_height = max(1.0, bbox_y2 - bbox_y1)
            long_extent = max(bbox_width, bbox_height)
            short_extent = min(bbox_width, bbox_height)
            # Derive grouping tolerances from the detected cue geometry. This
            # accommodates different image scales without fixed pixel cutoffs.
            angle_tolerance = float(np.degrees(np.arctan2(short_extent, long_extent)))
            collinear_tolerance = short_extent * 0.4

            def angular_distance(first: float, second: float) -> float:
                return abs((first - second + 90.0) % 180.0 - 90.0)

            best_cluster: list[dict[str, Any]] = []
            best_score = (-1.0, -1.0, -1)
            for seed in segments:
                seed_angle = seed["angle"]
                radians = np.radians(seed_angle)
                direction = (float(np.cos(radians)), float(np.sin(radians)))
                normal = (-direction[1], direction[0])
                seed_midpoint = seed["midpoint"]
                cluster = []
                for segment in segments:
                    if angular_distance(segment["angle"], seed_angle) > angle_tolerance:
                        continue
                    midpoint = segment["midpoint"]
                    perpendicular_distance = abs(
                        (midpoint[0] - seed_midpoint[0]) * normal[0]
                        + (midpoint[1] - seed_midpoint[1]) * normal[1]
                    )
                    if perpendicular_distance <= collinear_tolerance:
                        cluster.append(segment)

                # Score the unique projected span so overlapping Hough fragments
                # do not outweigh a coherent shaft visible across the bbox.
                intervals = []
                for segment in cluster:
                    projections = [
                        point[0] * direction[0] + point[1] * direction[1]
                        for point in (segment["start"], segment["end"])
                    ]
                    intervals.append((min(projections), max(projections)))
                intervals.sort()
                covered_length = 0.0
                if intervals:
                    interval_start, interval_end = intervals[0]
                    for next_start, next_end in intervals[1:]:
                        if next_start <= interval_end:
                            interval_end = max(interval_end, next_end)
                        else:
                            covered_length += interval_end - interval_start
                            interval_start, interval_end = next_start, next_end
                    covered_length += interval_end - interval_start
                score = (
                    covered_length,
                    sum(item["length"] for item in cluster),
                    len(cluster),
                )
                if score > best_score:
                    best_score = score
                    best_cluster = cluster

            if not best_cluster:
                logger.debug("No coherent cue line found, using bbox aspect ratio")
                return 0.0 if cue_detection.width > cue_detection.height else 90.0

            # Average line orientation as an undirected axis (modulo 180°), and
            # use segment midpoints to place the fitted axis before bbox clipping.
            doubled_angles = np.radians(
                [2.0 * segment["angle"] for segment in best_cluster]
            )
            weights = np.asarray([segment["length"] for segment in best_cluster])
            angle_rad = 0.5 * float(
                np.arctan2(
                    np.sum(np.sin(doubled_angles) * weights),
                    np.sum(np.cos(doubled_angles) * weights),
                )
            )
            direction = (float(np.cos(angle_rad)), float(np.sin(angle_rad)))
            total_weight = float(np.sum(weights))
            line_point = (
                sum(
                    segment["midpoint"][0] * segment["length"]
                    for segment in best_cluster
                )
                / total_weight,
                sum(
                    segment["midpoint"][1] * segment["length"]
                    for segment in best_cluster
                )
                / total_weight,
            )

            # Extend the fitted shaft axis to the model bbox edges, rather than
            # exposing a short Hough fragment to downstream cue rendering.
            intersections = []
            if abs(direction[0]) > np.finfo(float).eps:
                for edge_x in (bbox_x1, bbox_x2):
                    distance = (edge_x - line_point[0]) / direction[0]
                    edge_y = line_point[1] + distance * direction[1]
                    if bbox_y1 <= edge_y <= bbox_y2:
                        intersections.append((edge_x, edge_y))
            if abs(direction[1]) > np.finfo(float).eps:
                for edge_y in (bbox_y1, bbox_y2):
                    distance = (edge_y - line_point[1]) / direction[1]
                    edge_x = line_point[0] + distance * direction[0]
                    if bbox_x1 <= edge_x <= bbox_x2:
                        intersections.append((edge_x, edge_y))
            if len(intersections) < 2:
                logger.debug(
                    "Fitted cue line did not cross bbox, using bbox aspect ratio"
                )
                return 0.0 if cue_detection.width > cue_detection.height else 90.0

            line_end1, line_end2 = max(
                (
                    (first, second)
                    for index, first in enumerate(intersections)
                    for second in intersections[index + 1 :]
                ),
                key=lambda pair: float(
                    np.hypot(pair[1][0] - pair[0][0], pair[1][1] - pair[0][1])
                ),
            )
            cue_detection.line_end1 = line_end1
            cue_detection.line_end2 = line_end2
            cue_detection.line_center = (
                (line_end1[0] + line_end2[0]) / 2,
                (line_end1[1] + line_end2[1]) / 2,
            )
            angle_deg = float(
                np.degrees(
                    np.arctan2(
                        line_end2[1] - line_end1[1],
                        line_end2[0] - line_end1[0],
                    )
                )
                % 360.0
            )

            total_time = (time.perf_counter() - method_start) * 1000
            logger.debug(
                f"[PERF] _estimate_cue_angle timing: total={total_time:.1f}ms "
                f"(extract={extract_time:.1f}ms, gray={gray_time:.1f}ms, "
                f"blur={blur_time:.1f}ms, canny={canny_time:.1f}ms, "
                f"hough={hough_time:.1f}ms) | "
                f"angle={angle_deg:.1f}°, region_size={cue_region.shape[1]}x{cue_region.shape[0]}"
            )

            return float(angle_deg)

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

    def _run_inference(
        self,
        frame: NDArray[np.uint8],
        *,
        frame_key: Optional[Hashable] = None,
    ) -> list[Detection]:
        """Run YOLO inference on frame and convert results to Detection objects.

        Uses a single-entry cache only when the caller supplies an explicit key.
        The key must identify both the source frame and the image view (for example,
        full frame versus a crop). Calls without a key deliberately bypass caching.

        Args:
            frame: Input image in BGR format
            frame_key: Hashable source-frame/view identity. A key may be reused
                only while the pixels and view are unchanged.

        Returns:
            List of all detections
        """
        if not self.model_loaded:
            raise _ModelUnavailableError("inference backend is not loaded")

        if frame_key is not None:
            try:
                hash(frame_key)
            except TypeError:
                logger.warning("Unhashable frame_key; running inference uncached")
                frame_key = None

        cache_key = (
            (getattr(self, "_inference_model_generation", 0), frame_key)
            if frame_key is not None
            else None
        )

        # Keep the cache scoped to exactly one explicit frame/view. Holding the
        # lock through inference prevents concurrent ball/cue requests for that
        # same key from launching duplicate model calls. The optional audit span
        # distinguishes cache-lock wait from the detector's actual service time.
        profiler = getattr(self, "audit_profiler", None)
        if profiler is None:
            with self._cache_lock:
                return self._run_inference_locked(frame, cache_key)

        with profiler.span("detector_cache_lock_wait"):
            self._cache_lock.acquire()
        try:
            return self._run_inference_locked(frame, cache_key)
        finally:
            self._cache_lock.release()

    def _run_inference_locked(
        self,
        frame: NDArray[np.uint8],
        cache_key: tuple[int, Hashable] | None,
    ) -> list[Detection]:
        """Run a cache lookup and detector backend while owning its cache lock."""
        if cache_key is not None and cache_key in self._detection_cache:
            profiler = getattr(self, "audit_profiler", None)
            current_frame = profiler.current_frame if profiler is not None else None
            if current_frame is not None:
                current_frame.metadata["detector_cache_hits"] = (
                    current_frame.metadata.get("detector_cache_hits", 0) + 1
                )
            logger.debug("[PERF] Reusing inference for current frame/view")
            return list(self._detection_cache[cache_key])

        self._detection_cache.clear()
        profiler = getattr(self, "audit_profiler", None)
        if profiler is None:
            detections = (
                self._run_tpu_inference(frame)
                if self.model_format == ModelFormat.EDGETPU
                else self._run_standard_inference(frame)
            )
        else:
            with profiler.span("detector_backend"):
                detections = (
                    self._run_tpu_inference(frame)
                    if self.model_format == ModelFormat.EDGETPU
                    else self._run_standard_inference(frame)
                )

        if cache_key is not None:
            self._detection_cache[cache_key] = detections
        return detections

    def _run_standard_inference(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run standard YOLO inference (PyTorch, ONNX, TensorRT).

        Args:
            frame: Input image in BGR format

        Returns:
            List of all detections
        """
        import time

        start_time = time.perf_counter()
        inference_start = None
        profiler = getattr(self, "audit_profiler", None)
        failure_stage = DetectionFailureStage.INFERENCE

        try:
            if profiler is None:
                # Acquire lock for thread-safe model access during inference.
                with self._model_lock:
                    if not self.model_loaded or self.model is None:
                        raise _ModelUnavailableError("inference backend is not loaded")
                    inference_start = time.perf_counter()
                    results = self._invoke_standard_model(
                        self.model, frame, self.confidence, self.nms_threshold
                    )
            else:
                # Separate waiting for the shared model lock from the actual
                # backend call so contention is visible in audit reports.
                with profiler.span("detector_model_lock_wait"):
                    self._model_lock.acquire()
                try:
                    if not self.model_loaded or self.model is None:
                        raise _ModelUnavailableError("inference backend is not loaded")
                    inference_start = time.perf_counter()
                    with profiler.span("model_execute"):
                        results = self._invoke_standard_model(
                            self.model, frame, self.confidence, self.nms_threshold
                        )
                finally:
                    self._model_lock.release()

            inference_time = (time.perf_counter() - inference_start) * 1000  # ms

            # Parse results (outside lock - parsing doesn't need model access)
            failure_stage = DetectionFailureStage.DECODING
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
            total_time = (time.perf_counter() - start_time) * 1000  # ms
            self.stats["total_inferences"] += 1
            self.stats["total_detections"] += len(detections)

            # Update average inference time (exponential moving average)
            alpha = 0.1
            self.stats["avg_inference_time"] = (
                alpha * total_time + (1 - alpha) * self.stats["avg_inference_time"]
            )

            # Log detailed timing breakdown
            parse_time = total_time - inference_time if inference_start else 0
            logger.debug(
                f"YOLO: {len(detections)} detections | "
                f"inference={inference_time:.1f}ms, parse={parse_time:.1f}ms, total={total_time:.1f}ms"
            )

            return detections

        except _ModelUnavailableError:
            raise
        except _DetectionBackendFailure:
            raise
        except Exception as e:
            logger.error(
                "YOLO standard %s failed (%s)",
                failure_stage.value,
                type(e).__name__,
            )
            raise _DetectionBackendFailure(failure_stage) from e

    def _invoke_standard_model(
        self,
        model: Callable[..., Any],
        frame: NDArray[np.uint8],
        confidence: float,
        nms_threshold: float,
    ) -> Any:
        """Call Ultralytics with the configured device for PyTorch weights.

        Ultralytics selects a device when its predictor is first initialized.
        The detector's ``device`` setting must therefore be passed at inference
        time; logging it during model loading does not place the model there.
        Other formats retain their format-specific backend selection until that
        backend has explicit placement verification.
        """
        kwargs: dict[str, Any] = {
            "conf": confidence,
            "iou": nms_threshold,
            "verbose": False,
        }
        if self.model_format == ModelFormat.PYTORCH:
            kwargs["device"] = self.device

        return model(frame, **kwargs)

    def _get_effective_inference_device(self) -> str | None:
        """Return an observed PyTorch predictor device when Ultralytics exposes it.

        Keep the configured/requested device separate from runtime evidence. A
        value that cannot be observed remains unavailable rather than being
        inferred from the configuration or from provider availability.
        """
        if self.model_format != ModelFormat.PYTORCH or self.model is None:
            return None

        predictor = getattr(self.model, "predictor", None)
        predictor_device = getattr(predictor, "device", None)
        return str(predictor_device) if predictor_device is not None else None

    def _get_effective_inference_backend(self) -> str | None:
        """Return the verified inference backend family when it is observable.

        CoreML packages are loaded through Ultralytics' native CoreML path. This
        reports the backend family only; CoreML's actual CPU/GPU/Neural Engine
        partition remains unavailable unless the runtime exposes it.
        """
        if not self.model_loaded:
            return None
        if self.model_format == ModelFormat.COREML:
            return "coreml"
        if self.model_format == ModelFormat.PYTORCH:
            return "pytorch"
        return None

    def _run_tpu_inference(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run Edge TPU inference on frame.

        Args:
            frame: Input image in BGR format

        Returns:
            List of all detections
        """
        import time

        start_time = time.perf_counter()
        failure_stage = DetectionFailureStage.INFERENCE

        try:
            # Acquire lock for thread-safe TPU access
            with self._model_lock:
                if (
                    not self.model_loaded
                    or self.tpu_interpreter is None
                    or self.tpu_input_details is None
                    or self.tpu_output_details is None
                ):
                    raise _ModelUnavailableError("TPU inference backend is not loaded")

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
                outputs: list[NDArray[Any]] = []
                for output_detail in self.tpu_output_details:
                    output = self.tpu_interpreter.get_tensor(output_detail["index"])
                    outputs.append(output)

            # Parse TPU output to Detection objects
            failure_stage = DetectionFailureStage.DECODING
            detections = self._parse_tpu_output(
                outputs, frame.shape, (input_height, input_width)
            )

            # Update statistics
            inference_time = (time.perf_counter() - start_time) * 1000  # ms
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

        except _ModelUnavailableError:
            raise
        except _DetectionBackendFailure:
            raise
        except Exception as e:
            logger.error("TPU %s failed (%s)", failure_stage.value, type(e).__name__)
            raise _DetectionBackendFailure(failure_stage) from e

    def _parse_tpu_output(
        self,
        outputs: list[NDArray[Any]],
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
        detections: list[Detection] = []

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
            # Prediction counts are normally much larger than the feature
            # count. Keep the larger non-batch axis as predictions so both
            # [1, N, F] and [1, F, N] exports decode to rows of features.
            if output.shape[1] >= output.shape[2]:
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
        class_groups: dict[int, list[Detection]] = {}
        for det in detections:
            if det.class_id not in class_groups:
                class_groups[det.class_id] = []
            class_groups[det.class_id].append(det)

        # Apply NMS per class
        final_detections: list[Detection] = []
        for class_id, class_dets in class_groups.items():
            # OpenCV NMSBoxes expects (x, y, width, height), while Detection
            # stores (x1, y1, x2, y2). Passing corners as sizes makes large
            # image coordinates look like enormous, overlapping boxes.
            boxes = np.array(
                [
                    (
                        detection.bbox[0],
                        detection.bbox[1],
                        detection.bbox[2] - detection.bbox[0],
                        detection.bbox[3] - detection.bbox[1],
                    )
                    for detection in class_dets
                ]
            )
            scores = np.array([d.confidence for d in class_dets])

            # OpenCV NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), scores.tolist(), self.confidence, iou_threshold
            )

            # Extract kept detections
            if len(indices) > 0:
                normalized_indices = np.asarray(indices).reshape(-1)
                for idx in normalized_indices:
                    final_detections.append(class_dets[int(idx)])

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
            # ``device`` remains for API compatibility and means the configured
            # request. Only ``effective_device`` reports runtime placement, and
            # it stays null until the PyTorch predictor exposes that evidence.
            "device": self.device,
            "requested_device": self.device,
            "effective_device": self._get_effective_inference_device(),
            "effective_backend": self._get_effective_inference_backend(),
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
                loaded_model = cast("Optional[Callable[..., Any]]", self.model)
                if not self.model_loaded or loaded_model is None:
                    raise RuntimeError("Model loaded but not marked as available")

                # Test model with a dummy inference to ensure it works
                logger.info("Testing new model with dummy inference")
                test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self._invoke_standard_model(
                    loaded_model,
                    test_frame,
                    confidence=0.5,
                    nms_threshold=self.nms_threshold,
                )

                # Success! Log the change
                logger.info(
                    f"Model reloaded successfully: {previous_model_path} -> {new_model_path}"
                )
                logger.info(
                    f"Model format: {self.model_format.value if self.model_format else 'unknown'}"
                )

                # Cached detections are specific to the model that produced them.
                self._inference_model_generation = (
                    getattr(self, "_inference_model_generation", 0) + 1
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
            - enable_opencv_classification: bool (default False; heuristic exact-number guesses)
            - tpu_device_path: str (optional, for TPU)
            - min_ball_size: int (optional, minimum ball size in pixels, default 20)

    Returns:
        Configured YOLODetector instance

    Note:
        Uses conservative defaults from VisionModule: confidence=0.15 and
        enable_opencv_classification=False. Exact-number guesses remain opt-in
        because the configured detector model does not predict ball identities.
    """
    if config is None:
        config = {}

    return YOLODetector(
        model_path=model_path,
        device=config.get("device", "cpu"),
        confidence=config.get("confidence", 0.15),
        nms_threshold=config.get("nms_threshold", 0.45),
        auto_fallback=config.get("auto_fallback", True),
        enable_opencv_classification=config.get("enable_opencv_classification", False),
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
