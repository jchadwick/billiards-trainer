"""Image preprocessing pipeline for the vision module.

Provides comprehensive image preprocessing including color space conversion,
noise reduction, exposure correction, and optimization for detection algorithms.

Implements requirements:
- FR-VIS-006: Convert color spaces (BGR to HSV/LAB) for robust detection
- FR-VIS-007: Apply noise reduction and image smoothing
- FR-VIS-008: Perform automatic exposure and white balance correction
- FR-VIS-009: Crop to region of interest (ROI) for performance optimization
- FR-VIS-010: Handle varying lighting conditions adaptively
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

# Optional GPU acceleration
try:
    from .gpu_utils import get_gpu_accelerator

    _HAS_GPU_UTILS = True
except ImportError:
    _HAS_GPU_UTILS = False

    def get_gpu_accelerator():
        return None


logger = logging.getLogger(__name__)


class ColorSpace(Enum):
    """Supported color spaces."""

    BGR = "bgr"
    RGB = "rgb"
    HSV = "hsv"
    LAB = "lab"
    GRAY = "gray"
    YUV = "yuv"


class NoiseReductionMethod(Enum):
    """Noise reduction algorithms."""

    GAUSSIAN = "gaussian"
    BILATERAL = "bilateral"
    NON_LOCAL_MEANS = "non_local_means"
    MEDIAN = "median"


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""

    # Color space conversion
    target_color_space: ColorSpace = ColorSpace.HSV
    normalize_brightness: bool = True
    auto_white_balance: bool = True

    # Noise reduction
    noise_reduction_enabled: bool = True
    noise_method: NoiseReductionMethod = NoiseReductionMethod.GAUSSIAN
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    median_kernel_size: int = 5
    non_local_means_h: float = 10.0
    non_local_means_h_color: float = 10.0
    non_local_means_template_window_size: int = 7
    non_local_means_search_window_size: int = 21

    # Morphological operations
    morphology_enabled: bool = True
    morphology_kernel_size: int = 3
    morphology_iterations: int = 1

    # Exposure and contrast
    auto_exposure_correction: bool = True
    contrast_enhancement: bool = True
    gamma_correction: float = 1.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    contrast_alpha: float = 1.2
    contrast_beta: int = 10
    default_target_brightness: float = 128.0
    default_contrast_alpha: float = 1.5

    # Sharpening
    sharpening_enabled: bool = False
    sharpening_strength: float = 0.5

    # Performance optimization
    enable_gpu: bool = False
    resize_for_processing: bool = False
    processing_scale: float = 0.5

    # Debug settings
    debug_mode: bool = False
    save_intermediate_steps: bool = False


class ImagePreprocessor:
    """Image preprocessing for vision pipeline with comprehensive filtering.

    Features:
    - Multi-color space conversion
    - Adaptive noise reduction
    - Automatic exposure and white balance correction
    - Morphological operations
    - GPU acceleration support
    - Performance optimization with scaling
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        """Initialize preprocessor with configuration.

        Args:
            config: Configuration dictionary containing preprocessing parameters.
                   If None, uses default configuration.
        """
        # Convert config dict to structured config
        self.config = PreprocessingConfig(**config) if config else PreprocessingConfig()

        # Initialize GPU accelerator if enabled
        # NOTE: GPU is disabled by default to prevent OpenCL initialization hangs
        self.gpu = None
        if self.config.enable_gpu:
            logger.warning(
                "GPU acceleration requested but disabled by default to prevent initialization hangs"
            )
            logger.info(
                "GPU acceleration is opt-in only. To enable, modify gpu_utils.py to enable OpenCL"
            )
            logger.info("Using CPU-based processing for all operations")
            # Do not initialize GPU accelerator to avoid potential hangs
            # try:
            #     self.gpu = get_gpu_accelerator()
            #     if self.gpu.is_available():
            #         logger.info("GPU acceleration ENABLED for preprocessing")
            #         gpu_info = self.gpu.get_info()
            #         logger.info(f"GPU info: {gpu_info}")
            #     else:
            #         logger.warning("GPU acceleration requested but not available - using CPU")
            #         self.gpu = None
            # except Exception as e:
            #     logger.error(f"Failed to initialize GPU acceleration: {e}")
            #     self.gpu = None

        # Initialize processing pipeline
        self._initialize_pipeline()

        # Statistics tracking
        self.stats = {
            "frames_processed": 0,
            "avg_processing_time": 0.0,
            "last_brightness": 0.0,
            "last_contrast": 0.0,
            "gpu_enabled": self.gpu is not None,
        }

        # Debug storage
        self.debug_images: list[tuple[str, NDArray[np.uint8]]] = []

        target_space = self.config.target_color_space
        logger.info(f"Image preprocessor initialized with target: {target_space}")

    def _initialize_pipeline(self) -> None:
        """Initialize the preprocessing pipeline components."""
        # Create morphological kernel
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morphology_kernel_size, self.config.morphology_kernel_size),
        )

        # Initialize CLAHE for adaptive histogram equalization
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )

        # Sharpening kernel
        if self.config.sharpening_enabled:
            self.sharpen_kernel = (
                np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                * self.config.sharpening_strength
            )

        logger.debug("Preprocessing pipeline components initialized")

    def process(self, frame: NDArray[np.uint8]) -> NDArray[np.float64]:
        """Apply complete preprocessing pipeline to frame.

        Args:
            frame: Input frame in BGR format

        Returns:
            Preprocessed frame in target color space
        """
        if frame is None or frame.size == 0:
            return frame

        start_time = cv2.getTickCount()

        try:
            processed_frame = frame.copy()

            # Step 1: Resize for processing if enabled
            original_size = None
            if (
                self.config.resize_for_processing
                and self.config.processing_scale != 1.0
            ):
                original_size = (frame.shape[1], frame.shape[0])
                new_width = int(frame.shape[1] * self.config.processing_scale)
                new_height = int(frame.shape[0] * self.config.processing_scale)
                # Use GPU if available
                if self.gpu:
                    processed_frame = self.gpu.resize(
                        processed_frame, (new_width, new_height)
                    )
                else:
                    processed_frame = cv2.resize(
                        processed_frame, (new_width, new_height)
                    )

                if self.config.debug_mode:
                    self.debug_images.append(("resized", processed_frame.copy()))

            # Step 2: Auto white balance
            if self.config.auto_white_balance:
                processed_frame = self._apply_white_balance(processed_frame)

                if self.config.debug_mode:
                    self.debug_images.append(("white_balanced", processed_frame.copy()))

            # Step 3: Exposure and contrast correction
            if self.config.auto_exposure_correction or self.config.contrast_enhancement:
                processed_frame = self._correct_exposure_contrast(processed_frame)

                if self.config.debug_mode:
                    self.debug_images.append(
                        ("exposure_corrected", processed_frame.copy())
                    )

            # Step 4: Noise reduction
            if self.config.noise_reduction_enabled:
                processed_frame = self._apply_noise_reduction(processed_frame)

                if self.config.debug_mode:
                    self.debug_images.append(("noise_reduced", processed_frame.copy()))

            # Step 5: Sharpening
            if self.config.sharpening_enabled:
                processed_frame = self._apply_sharpening(processed_frame)

                if self.config.debug_mode:
                    self.debug_images.append(("sharpened", processed_frame.copy()))

            # Step 6: Color space conversion
            processed_frame = self._convert_color_space(
                processed_frame, self.config.target_color_space
            )

            if self.config.debug_mode:
                self.debug_images.append(("color_converted", processed_frame.copy()))

            # Step 7: Morphological operations (if still in appropriate format)
            if self.config.morphology_enabled and len(processed_frame.shape) == 3:
                processed_frame = self._apply_morphology(processed_frame)

                if self.config.debug_mode:
                    self.debug_images.append(("morphology", processed_frame.copy()))

            # Step 8: Resize back to original size if needed
            if original_size is not None:
                # Use GPU if available
                if self.gpu:
                    processed_frame = self.gpu.resize(processed_frame, original_size)
                else:
                    processed_frame = cv2.resize(processed_frame, original_size)

            # Update statistics
            processing_time = (
                (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
            )
            self._update_stats(processing_time)

            return processed_frame

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return frame  # Return original frame on error

    def convert_color_space(
        self, frame: NDArray[np.uint8], target: str = "HSV"
    ) -> NDArray[np.float64]:
        """Convert frame to target color space.

        Args:
            frame: Input frame
            target: Target color space ("HSV", "LAB", "RGB", "GRAY", etc.)

        Returns:
            Frame in target color space
        """
        try:
            color_space = ColorSpace(target.lower())
            return self._convert_color_space(frame, color_space)
        except ValueError:
            logger.warning(f"Unsupported color space: {target}")
            return frame

    def apply_noise_reduction(
        self, frame: NDArray[np.uint8], method: Optional[str] = None
    ) -> NDArray[np.float64]:
        """Apply noise reduction filters.

        Args:
            frame: Input frame
            method: Noise reduction method ("gaussian", "bilateral", "median", etc.)

        Returns:
            Noise-reduced frame
        """
        if method:
            try:
                noise_method = NoiseReductionMethod(method.lower())
                return self._apply_specific_noise_reduction(frame, noise_method)
            except ValueError:
                logger.warning(f"Unsupported noise reduction method: {method}")

        return self._apply_noise_reduction(frame)

    def normalize_brightness(
        self,
        frame: NDArray[np.uint8],
        target_brightness: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """Normalize frame brightness to target level.

        Args:
            frame: Input frame
            target_brightness: Target average brightness (0-255).
                             If None, uses config default.

        Returns:
            Brightness-normalized frame
        """
        if target_brightness is None:
            target_brightness = self.config.default_target_brightness
        if len(frame.shape) == 3:
            # Convert to grayscale for brightness calculation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        current_brightness = np.mean(gray)

        if current_brightness > 0:
            brightness_factor = target_brightness / current_brightness
            normalized = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)

            self.stats["last_brightness"] = current_brightness
            return normalized

        return frame

    def enhance_contrast(
        self, frame: NDArray[np.uint8], alpha: Optional[float] = None
    ) -> NDArray[np.float64]:
        """Enhance frame contrast.

        Args:
            frame: Input frame
            alpha: Contrast multiplier. If None, uses config default.

        Returns:
            Contrast-enhanced frame
        """
        if alpha is None:
            alpha = self.config.default_contrast_alpha
        enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
        return enhanced

    def apply_gamma_correction(
        self, frame: NDArray[np.uint8], gamma: float = 1.0
    ) -> NDArray[np.float64]:
        """Apply gamma correction to frame.

        Args:
            frame: Input frame
            gamma: Gamma value (1.0 = no change, <1.0 = brighter, >1.0 = darker)

        Returns:
            Gamma-corrected frame
        """
        if gamma == 1.0:
            return frame

        # Build lookup table
        lookup_table = np.array(
            [(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8
        )

        return cv2.LUT(frame, lookup_table)

    def get_debug_images(self) -> list:
        """Get debug images from last preprocessing run."""
        return self.debug_images

    def clear_debug_images(self) -> None:
        """Clear debug image storage."""
        self.debug_images.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get preprocessing statistics."""
        return self.stats.copy()

    def reduce_noise(
        self, frame: NDArray[np.uint8], method: Optional[str] = None
    ) -> NDArray[np.uint8]:
        """Apply noise reduction to frame.

        Alias for apply_noise_reduction for backward compatibility.

        Args:
            frame: Input frame
            method: Noise reduction method (optional)

        Returns:
            Noise-reduced frame
        """
        result = self.apply_noise_reduction(frame, method)
        return result.astype(np.uint8) if result.dtype != np.uint8 else result

    def correct_lighting(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Correct uneven lighting in frame.

        Applies exposure and contrast correction to normalize lighting.

        Args:
            frame: Input frame

        Returns:
            Lighting-corrected frame
        """
        result = self._correct_exposure_contrast(frame)
        return result.astype(np.uint8) if result.dtype != np.uint8 else result

    def extract_roi(
        self,
        frame: NDArray[np.uint8],
        center: tuple[int, int],
        size: tuple[int, int],
    ) -> NDArray[np.uint8]:
        """Extract region of interest from frame.

        Args:
            frame: Input frame
            center: Center point (x, y) of ROI
            size: Size (width, height) of ROI

        Returns:
            Extracted ROI
        """
        x, y = center
        width, height = size

        # Calculate bounds
        x1 = max(0, x - width // 2)
        y1 = max(0, y - height // 2)
        x2 = min(frame.shape[1], x + width // 2)
        y2 = min(frame.shape[0], y + height // 2)

        # Extract ROI
        roi = frame[y1:y2, x1:x2]

        return roi

    # Private helper methods

    def _convert_color_space(
        self, frame: NDArray[np.uint8], target: ColorSpace
    ) -> NDArray[np.float64]:
        """Convert frame to target color space."""
        if len(frame.shape) == 2:  # Already grayscale
            if target == ColorSpace.GRAY:
                return frame
            elif target == ColorSpace.BGR:
                code = cv2.COLOR_GRAY2BGR
                return (
                    self.gpu.cvt_color(frame, code)
                    if self.gpu
                    else cv2.cvtColor(frame, code)
                )
            elif target == ColorSpace.RGB:
                code = cv2.COLOR_GRAY2RGB
                return (
                    self.gpu.cvt_color(frame, code)
                    if self.gpu
                    else cv2.cvtColor(frame, code)
                )
            elif target == ColorSpace.HSV:
                bgr = (
                    self.gpu.cvt_color(frame, cv2.COLOR_GRAY2BGR)
                    if self.gpu
                    else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                )
                return (
                    self.gpu.cvt_color(bgr, cv2.COLOR_BGR2HSV)
                    if self.gpu
                    else cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                )

        # Convert from BGR (OpenCV default)
        conversion_map = {
            ColorSpace.RGB: cv2.COLOR_BGR2RGB,
            ColorSpace.HSV: cv2.COLOR_BGR2HSV,
            ColorSpace.LAB: cv2.COLOR_BGR2LAB,
            ColorSpace.GRAY: cv2.COLOR_BGR2GRAY,
            ColorSpace.YUV: cv2.COLOR_BGR2YUV,
            ColorSpace.BGR: None,  # No conversion needed
        }

        if target == ColorSpace.BGR:
            return frame

        conversion_code = conversion_map.get(target)
        if conversion_code is not None:
            # Use GPU if available
            if self.gpu:
                return self.gpu.cvt_color(frame, conversion_code)
            else:
                return cv2.cvtColor(frame, conversion_code)

        logger.warning(f"Conversion to {target} not implemented")
        return frame

    def _apply_white_balance(self, frame: NDArray[np.uint8]) -> NDArray[np.float64]:
        """Apply automatic white balance correction."""
        try:
            # Simple white balance using gray world assumption
            result = cv2.xphoto.createSimpleWB()
            return result.balanceWhite(frame)
        except AttributeError:
            # Fallback method if opencv-contrib not available
            return self._simple_white_balance(frame)

    def _simple_white_balance(self, frame: NDArray[np.uint8]) -> NDArray[np.float64]:
        """Simple white balance implementation."""
        # Calculate average for each channel
        avg_b = np.mean(frame[:, :, 0])
        avg_g = np.mean(frame[:, :, 1])
        avg_r = np.mean(frame[:, :, 2])

        # Find the maximum average (reference)
        max_avg = max(avg_b, avg_g, avg_r)

        # Calculate scaling factors
        scale_b = max_avg / avg_b if avg_b > 0 else 1.0
        scale_g = max_avg / avg_g if avg_g > 0 else 1.0
        scale_r = max_avg / avg_r if avg_r > 0 else 1.0

        # Apply scaling
        balanced = frame.copy().astype(np.float32)
        balanced[:, :, 0] *= scale_b
        balanced[:, :, 1] *= scale_g
        balanced[:, :, 2] *= scale_r

        # Clip and convert back
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)

        return balanced

    def _correct_exposure_contrast(
        self, frame: NDArray[np.uint8]
    ) -> NDArray[np.float64]:
        """Apply exposure and contrast correction."""
        corrected = frame.copy()

        if self.config.auto_exposure_correction:
            # Convert to LAB color space for luminance processing
            lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)

            # Apply CLAHE to L channel
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])

            # Convert back to BGR
            corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if self.config.contrast_enhancement:
            # Simple contrast enhancement
            corrected = cv2.convertScaleAbs(
                corrected,
                alpha=self.config.contrast_alpha,
                beta=self.config.contrast_beta,
            )

        if self.config.gamma_correction != 1.0:
            corrected = self.apply_gamma_correction(
                corrected, self.config.gamma_correction
            )

        return corrected

    def _apply_noise_reduction(self, frame: NDArray[np.uint8]) -> NDArray[np.float64]:
        """Apply configured noise reduction method."""
        return self._apply_specific_noise_reduction(frame, self.config.noise_method)

    def _apply_specific_noise_reduction(
        self, frame: NDArray[np.uint8], method: NoiseReductionMethod
    ) -> NDArray[np.float64]:
        """Apply specific noise reduction method."""
        if method == NoiseReductionMethod.GAUSSIAN:
            ksize = (self.config.gaussian_kernel_size, self.config.gaussian_kernel_size)
            sigma = self.config.gaussian_sigma
            if self.gpu:
                return self.gpu.gaussian_blur(frame, ksize, sigma)
            else:
                return cv2.GaussianBlur(frame, ksize, sigma)

        elif method == NoiseReductionMethod.BILATERAL:
            if self.gpu:
                return self.gpu.bilateral_filter(
                    frame,
                    self.config.bilateral_d,
                    self.config.bilateral_sigma_color,
                    self.config.bilateral_sigma_space,
                )
            else:
                return cv2.bilateralFilter(
                    frame,
                    self.config.bilateral_d,
                    self.config.bilateral_sigma_color,
                    self.config.bilateral_sigma_space,
                )

        elif method == NoiseReductionMethod.MEDIAN:
            if self.gpu:
                return self.gpu.median_blur(frame, self.config.median_kernel_size)
            else:
                return cv2.medianBlur(frame, self.config.median_kernel_size)

        elif method == NoiseReductionMethod.NON_LOCAL_MEANS:
            # Non-local means doesn't have GPU support in OpenCV
            if len(frame.shape) == 3:
                return cv2.fastNlMeansDenoisingColored(
                    frame,
                    None,
                    self.config.non_local_means_h,
                    self.config.non_local_means_h_color,
                    self.config.non_local_means_template_window_size,
                    self.config.non_local_means_search_window_size,
                )
            else:
                return cv2.fastNlMeansDenoising(
                    frame,
                    None,
                    self.config.non_local_means_h,
                    self.config.non_local_means_template_window_size,
                    self.config.non_local_means_search_window_size,
                )

        else:
            logger.warning(f"Unknown noise reduction method: {method}")
            return frame

    def _apply_sharpening(self, frame: NDArray[np.uint8]) -> NDArray[np.float64]:
        """Apply sharpening filter."""
        if hasattr(self, "sharpen_kernel"):
            if self.gpu:
                return self.gpu.filter_2d(frame, -1, self.sharpen_kernel)
            else:
                return cv2.filter2D(frame, -1, self.sharpen_kernel)
        return frame

    def _apply_morphology(self, frame: NDArray[np.uint8]) -> NDArray[np.float64]:
        """Apply morphological operations."""
        # Apply closing to fill small gaps
        if self.gpu:
            closed = self.gpu.morphology_ex(
                frame,
                cv2.MORPH_CLOSE,
                self.morph_kernel,
                iterations=self.config.morphology_iterations,
            )
        else:
            closed = cv2.morphologyEx(
                frame,
                cv2.MORPH_CLOSE,
                self.morph_kernel,
                iterations=self.config.morphology_iterations,
            )

        # Apply opening to remove small noise
        if self.gpu:
            opened = self.gpu.morphology_ex(
                closed,
                cv2.MORPH_OPEN,
                self.morph_kernel,
                iterations=self.config.morphology_iterations,
            )
        else:
            opened = cv2.morphologyEx(
                closed,
                cv2.MORPH_OPEN,
                self.morph_kernel,
                iterations=self.config.morphology_iterations,
            )

        return opened

    def _update_stats(self, processing_time: float) -> None:
        """Update processing statistics."""
        self.stats["frames_processed"] += 1

        # Update rolling average
        total_frames = self.stats["frames_processed"]
        self.stats["avg_processing_time"] = (
            self.stats["avg_processing_time"] * (total_frames - 1) + processing_time
        ) / total_frames


# Alias for backward compatibility with tests
FramePreprocessor = ImagePreprocessor
