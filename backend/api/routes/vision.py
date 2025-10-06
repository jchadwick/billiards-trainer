"""Vision module API routes for model management and vision operations."""

import asyncio
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from ...vision.detection.yolo_detector import ModelValidationError, YOLODetector
from ..dependencies import ApplicationState, get_app_state
from ..models.common import ErrorCode, create_error_response
from ..models.vision_models import ModelInfoResponse, ModelUploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision"])


# Request/Response Models for model reload
class ModelReloadRequest(BaseModel):
    """Request model for reloading YOLO model."""

    model_path: str = Field(
        ...,
        description="Absolute path to YOLO model file (.pt, .onnx, or .engine)",
        examples=["/path/to/model.onnx", "/opt/models/billiards_yolo.pt"],
    )


class ModelReloadResponse(BaseModel):
    """Response for model reload operation."""

    success: bool = Field(..., description="Whether reload was successful")
    message: str = Field(..., description="Status message")
    model_path: str = Field(..., description="Path to loaded model")
    previous_model: Optional[str] = Field(None, description="Previous model path")
    model_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Updated model information",
    )


# Models directory path (relative to project root)
MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"


@router.post(
    "/model/upload",
    response_model=ModelUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload custom ONNX model",
    description="""
    Upload a custom ONNX model for ball/cue/table detection.

    The model will be validated for:
    - Valid ONNX format
    - Compatible input/output shapes
    - Successful inference test

    The uploaded model will be saved with a timestamped version name.

    Requirements:
    - File must be .onnx format
    - Model should be YOLOv8-compatible
    - Input shape: [batch, 3, height, width] (typically 640x640)
    - Output: YOLO detection format
    """,
)
async def upload_model(
    file: UploadFile = File(..., description="ONNX model file to upload"),
    run_inference_test: bool = True,
) -> ModelUploadResponse:
    """Upload and validate a custom ONNX detection model.

    Args:
        file: Uploaded ONNX model file
        run_inference_test: Whether to run inference test (default: True)

    Returns:
        ModelUploadResponse with validation results

    Raises:
        HTTPException: If upload or validation fails
    """
    try:
        # Validate file extension
        if not file.filename or not file.filename.endswith(".onnx"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only .onnx files are supported",
            )

        # Create models directory if it doesn't exist
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Create temporary file for validation
        temp_file = None
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as temp_file:
                temp_path = Path(temp_file.name)
                # Read uploaded file in chunks
                content = await file.read()
                temp_file.write(content)
                temp_file.flush()

            logger.info(f"Uploaded model saved to temporary file: {temp_path}")

            # Validate ONNX model
            logger.info("Validating ONNX model...")
            try:
                validation_results = YOLODetector.validate_onnx_model(str(temp_path))
            except ModelValidationError as e:
                logger.error(f"Model validation failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model validation failed: {str(e)}",
                )

            # Run inference test if requested
            inference_test_results = None
            if run_inference_test:
                logger.info("Running inference test...")
                try:
                    inference_test_results = await asyncio.to_thread(
                        YOLODetector.test_model_inference, str(temp_path), "cpu"
                    )

                    if not inference_test_results.get("success"):
                        error_msg = inference_test_results.get(
                            "error", "Unknown inference error"
                        )
                        logger.error(f"Inference test failed: {error_msg}")
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Model inference test failed: {error_msg}",
                        )
                except Exception as e:
                    logger.error(f"Inference test failed: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Model inference test failed: {str(e)}",
                    )

            # Generate versioned filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(file.filename).stem
            version_name = f"{base_name}_{timestamp}.onnx"
            final_path = MODELS_DIR / version_name

            # Move validated model to models directory
            shutil.copy(temp_path, final_path)
            logger.info(f"Model saved to: {final_path}")

            # Clean up temporary file
            temp_path.unlink()

            return ModelUploadResponse(
                success=True,
                message="Model uploaded and validated successfully",
                model_path=str(final_path),
                model_name=version_name,
                validation_results=validation_results,
                inference_test=inference_test_results,
            )

        except HTTPException:
            # Clean up temp file on error
            if temp_file and temp_path.exists():
                temp_path.unlink()
            raise
        except Exception as e:
            # Clean up temp file on error
            if temp_file and temp_path.exists():
                temp_path.unlink()
            logger.error(f"Unexpected error uploading model: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload model: {str(e)}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling model upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get model information",
    description="Get information about the currently loaded model and available models",
)
async def get_model_info() -> ModelInfoResponse:
    """Get information about the current YOLO model and available models.

    Returns:
        ModelInfoResponse with model information
    """
    try:
        app_state = get_app_state()

        # Get current model info from vision module
        model_info = {}
        if app_state.vision_module and hasattr(app_state.vision_module, "detector"):
            detector = app_state.vision_module.detector
            if hasattr(detector, "get_model_info"):
                model_info = detector.get_model_info()

        # List available models in models directory
        available_models = []
        if MODELS_DIR.exists():
            available_models = [
                f.name for f in MODELS_DIR.iterdir() if f.suffix == ".onnx"
            ]

        return ModelInfoResponse(
            success=True,
            model_info=model_info,
            available_models=available_models,
        )

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}",
        )


@router.post(
    "/model/load",
    response_model=dict,
    summary="Load a model",
    description="Load a specific model from the models directory",
)
async def load_model(model_name: str) -> dict[str, Any]:
    """Load a specific model by name.

    Args:
        model_name: Name of the model file to load

    Returns:
        Response with load status

    Raises:
        HTTPException: If model loading fails
    """
    try:
        app_state = get_app_state()

        # Validate model exists
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_name}",
            )

        # Validate it's an ONNX file
        if model_path.suffix != ".onnx":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only .onnx models can be loaded",
            )

        # Load model into vision module detector
        if not app_state.vision_module:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vision module not available",
            )

        if not hasattr(app_state.vision_module, "detector"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Detector not available in vision module",
            )

        detector = app_state.vision_module.detector

        # Reload model with new path
        success = await asyncio.to_thread(detector.reload_model, str(model_path))

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {model_name}",
            )

        return {
            "success": True,
            "message": f"Model loaded successfully: {model_name}",
            "model_name": model_name,
            "model_path": str(model_path),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )


@router.post(
    "/model/reload",
    response_model=ModelReloadResponse,
    summary="Reload YOLO model with hot-swapping",
    description="""
    Reload the YOLO detection model without restarting the system.

    Implements FR-VIS-058: Support model hot-swapping without system restart.

    This endpoint allows you to:
    - Load a new YOLO model without restarting the system
    - Validate model before loading
    - Fallback to previous model if new model fails
    - Get detailed information about the loaded model

    The model reload is thread-safe and will not interrupt ongoing detection operations.

    Requirements:
    - Model path must be absolute
    - Model file must exist and be accessible
    - Model format must be .pt, .onnx, or .engine
    - Model must be compatible with YOLO detection
    """,
)
async def reload_yolo_model(
    request: ModelReloadRequest,
    app_state: ApplicationState = Depends(get_app_state),
) -> ModelReloadResponse:
    """Reload YOLO detection model with thread-safe hot-swapping.

    Args:
        request: Model reload request with model path
        app_state: Application state (injected)

    Returns:
        ModelReloadResponse with reload status and model information

    Raises:
        HTTPException: If vision module is unavailable or reload fails
    """
    try:
        # Get vision module
        if not app_state.vision_module:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=create_error_response(
                    "Vision Module Unavailable",
                    "Vision module is not initialized or not available",
                    ErrorCode.SYS_MODULE_UNAVAILABLE,
                    {"module": "vision"},
                ),
            )

        vision_module = app_state.vision_module

        # Check if vision module has detector
        if not hasattr(vision_module, "detector") or vision_module.detector is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=create_error_response(
                    "No Detector Available",
                    "Vision module does not have a YOLO detector configured",
                    ErrorCode.VAL_INVALID_CONFIGURATION,
                    {"hint": "Vision module may be using OpenCV-only detection"},
                ),
            )

        detector = vision_module.detector

        # Check if detector supports hot-swapping (has reload_model method)
        if not hasattr(detector, "reload_model"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=create_error_response(
                    "Detector Does Not Support Hot-Swapping",
                    "Current detector does not support model reloading",
                    ErrorCode.VAL_UNSUPPORTED_OPERATION,
                    {"detector_type": type(detector).__name__},
                ),
            )

        # Validate model path
        model_path = Path(request.model_path)

        if not model_path.is_absolute():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=create_error_response(
                    "Invalid Model Path",
                    "Model path must be an absolute path",
                    ErrorCode.VALIDATION_INVALID_FORMAT,
                    {"provided_path": request.model_path},
                ),
            )

        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    "Model File Not Found",
                    f"Model file does not exist: {request.model_path}",
                    ErrorCode.RES_NOT_FOUND,
                    {"model_path": request.model_path},
                ),
            )

        # Validate model format
        valid_extensions = [".pt", ".onnx", ".engine"]
        if model_path.suffix.lower() not in valid_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=create_error_response(
                    "Invalid Model Format",
                    f"Model must be one of: {', '.join(valid_extensions)}",
                    ErrorCode.VALIDATION_INVALID_FORMAT,
                    {
                        "provided_extension": model_path.suffix,
                        "valid_extensions": valid_extensions,
                    },
                ),
            )

        # Get current model info before reload
        previous_model_info = (
            detector.get_model_info() if hasattr(detector, "get_model_info") else {}
        )
        previous_model_path = previous_model_info.get("model_path")

        logger.info(f"Reloading model: {previous_model_path} -> {request.model_path}")

        # Perform model reload (run in thread to avoid blocking)
        success = await asyncio.to_thread(detector.reload_model, str(model_path))

        if not success:
            # Reload failed but detector handled it gracefully
            current_model_info = (
                detector.get_model_info() if hasattr(detector, "get_model_info") else {}
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=create_error_response(
                    "Model Reload Failed",
                    "Failed to reload model. Previous model may still be active.",
                    ErrorCode.SYS_INTERNAL_ERROR,
                    {
                        "requested_model": request.model_path,
                        "current_model": current_model_info.get("model_path"),
                        "fallback_mode": current_model_info.get("fallback_mode"),
                    },
                ),
            )

        # Get updated model info
        new_model_info = (
            detector.get_model_info() if hasattr(detector, "get_model_info") else {}
        )

        logger.info(f"Model reloaded successfully: {request.model_path}")

        return ModelReloadResponse(
            success=True,
            message=f"Model reloaded successfully from {request.model_path}",
            model_path=str(model_path),
            previous_model=previous_model_path,
            model_info=new_model_info,
        )

    except HTTPException:
        raise
    except ModelValidationError as e:
        logger.error(f"Model validation error during reload: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=create_error_response(
                "Model Validation Failed",
                str(e),
                ErrorCode.VAL_INVALID_CONFIGURATION,
                {"model_path": request.model_path, "error": str(e)},
            ),
        )
    except Exception as e:
        logger.error(f"Unexpected error during model reload: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                "Model Reload Error",
                "An unexpected error occurred during model reload",
                ErrorCode.SYS_INTERNAL_ERROR,
                {"error": str(e), "model_path": request.model_path},
            ),
        )
