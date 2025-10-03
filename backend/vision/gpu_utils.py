"""GPU acceleration utilities for video processing using VAAPI and OpenCL.

This module provides hardware-accelerated video processing operations
leveraging Intel VAAPI for GPU acceleration.
"""

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class GPUAccelerator:
    """Manages GPU acceleration for OpenCV operations using VAAPI and OpenCL.

    Features:
    - Hardware-accelerated video decoding via VAAPI
    - GPU-accelerated image processing via OpenCL
    - Automatic fallback to CPU when GPU unavailable
    - Memory management for GPU operations
    """

    def __init__(self):
        """Initialize GPU accelerator and check for hardware support."""
        self._gpu_available = False
        self._opencl_available = False
        self._vaapi_available = False
        self._initialized = False

        self._check_gpu_support()

    def _check_gpu_support(self) -> None:
        """Check for GPU hardware acceleration support."""
        try:
            # IMPORTANT: Disable OpenCL by default to prevent initialization hangs
            # OpenCL device initialization can hang indefinitely when Intel iHD VAAPI
            # driver initialization blocks. Only enable if explicitly requested.
            logger.info(
                "GPU acceleration disabled by default to prevent OpenCL initialization hangs"
            )

            # Explicitly disable OpenCL
            cv2.ocl.setUseOpenCL(False)
            self._opencl_available = False
            logger.info("OpenCL explicitly disabled to prevent device query deadlock")

            # Check for VAAPI support via environment (safe to check)
            libva_driver = os.environ.get("LIBVA_DRIVER_NAME", "")
            if libva_driver:
                self._vaapi_available = True
                logger.info(
                    f"VAAPI driver detected: {libva_driver} (using CPU-side operations)"
                )
            else:
                self._vaapi_available = False

            # GPU is considered unavailable since OpenCL is disabled
            self._gpu_available = False
            logger.warning("GPU acceleration is DISABLED by default - will use CPU")
            logger.info(
                "To enable GPU acceleration, set enable_opencl=True in configuration"
            )

            self._initialized = True

        except Exception as e:
            logger.error(f"Error checking GPU support: {e}", exc_info=True)
            self._gpu_available = False
            self._opencl_available = False
            self._vaapi_available = False

    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self._gpu_available

    def is_opencl_available(self) -> bool:
        """Check if OpenCL acceleration is available."""
        return self._opencl_available

    def is_vaapi_available(self) -> bool:
        """Check if VAAPI acceleration is available."""
        return self._vaapi_available

    def resize(
        self,
        frame: NDArray[np.uint8],
        size: tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
    ) -> NDArray[np.uint8]:
        """GPU-accelerated image resize.

        Args:
            frame: Input image
            size: Target size (width, height)
            interpolation: Interpolation method

        Returns:
            Resized image
        """
        if not self._opencl_available:
            return cv2.resize(frame, size, interpolation=interpolation)

        try:
            # Upload to GPU
            umat = cv2.UMat(frame)
            # Resize on GPU
            resized = cv2.resize(umat, size, interpolation=interpolation)
            # Download from GPU
            return resized.get()
        except Exception as e:
            logger.warning(f"GPU resize failed, falling back to CPU: {e}")
            return cv2.resize(frame, size, interpolation=interpolation)

    def cvt_color(self, frame: NDArray[np.uint8], code: int) -> NDArray[np.uint8]:
        """GPU-accelerated color space conversion.

        Args:
            frame: Input image
            code: OpenCV color conversion code

        Returns:
            Converted image
        """
        if not self._opencl_available:
            return cv2.cvtColor(frame, code)

        try:
            # Upload to GPU
            umat = cv2.UMat(frame)
            # Convert on GPU
            converted = cv2.cvtColor(umat, code)
            # Download from GPU
            return converted.get()
        except Exception as e:
            logger.warning(f"GPU color conversion failed, falling back to CPU: {e}")
            return cv2.cvtColor(frame, code)

    def gaussian_blur(
        self, frame: NDArray[np.uint8], ksize: tuple[int, int], sigma: float
    ) -> NDArray[np.uint8]:
        """GPU-accelerated Gaussian blur.

        Args:
            frame: Input image
            ksize: Kernel size (width, height)
            sigma: Gaussian sigma

        Returns:
            Blurred image
        """
        if not self._opencl_available:
            return cv2.GaussianBlur(frame, ksize, sigma)

        try:
            # Upload to GPU
            umat = cv2.UMat(frame)
            # Blur on GPU
            blurred = cv2.GaussianBlur(umat, ksize, sigma)
            # Download from GPU
            return blurred.get()
        except Exception as e:
            logger.warning(f"GPU Gaussian blur failed, falling back to CPU: {e}")
            return cv2.GaussianBlur(frame, ksize, sigma)

    def bilateral_filter(
        self, frame: NDArray[np.uint8], d: int, sigma_color: float, sigma_space: float
    ) -> NDArray[np.uint8]:
        """GPU-accelerated bilateral filter.

        Args:
            frame: Input image
            d: Diameter of pixel neighborhood
            sigma_color: Color space sigma
            sigma_space: Coordinate space sigma

        Returns:
            Filtered image
        """
        if not self._opencl_available:
            return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)

        try:
            # Upload to GPU
            umat = cv2.UMat(frame)
            # Filter on GPU
            filtered = cv2.bilateralFilter(umat, d, sigma_color, sigma_space)
            # Download from GPU
            return filtered.get()
        except Exception as e:
            logger.warning(f"GPU bilateral filter failed, falling back to CPU: {e}")
            return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)

    def median_blur(self, frame: NDArray[np.uint8], ksize: int) -> NDArray[np.uint8]:
        """GPU-accelerated median blur.

        Args:
            frame: Input image
            ksize: Kernel size (must be odd)

        Returns:
            Blurred image
        """
        if not self._opencl_available:
            return cv2.medianBlur(frame, ksize)

        try:
            # Upload to GPU
            umat = cv2.UMat(frame)
            # Blur on GPU
            blurred = cv2.medianBlur(umat, ksize)
            # Download from GPU
            return blurred.get()
        except Exception as e:
            logger.warning(f"GPU median blur failed, falling back to CPU: {e}")
            return cv2.medianBlur(frame, ksize)

    def morphology_ex(
        self,
        frame: NDArray[np.uint8],
        op: int,
        kernel: NDArray[np.uint8],
        iterations: int = 1,
    ) -> NDArray[np.uint8]:
        """GPU-accelerated morphological operations.

        Args:
            frame: Input image
            op: Morphology operation type
            kernel: Structuring element
            iterations: Number of iterations

        Returns:
            Processed image
        """
        if not self._opencl_available:
            return cv2.morphologyEx(frame, op, kernel, iterations=iterations)

        try:
            # Upload to GPU
            umat = cv2.UMat(frame)
            kernel_umat = cv2.UMat(kernel)
            # Process on GPU
            result = cv2.morphologyEx(umat, op, kernel_umat, iterations=iterations)
            # Download from GPU
            return result.get()
        except Exception as e:
            logger.warning(f"GPU morphology failed, falling back to CPU: {e}")
            return cv2.morphologyEx(frame, op, kernel, iterations=iterations)

    def filter_2d(
        self, frame: NDArray[np.uint8], ddepth: int, kernel: NDArray[np.float32]
    ) -> NDArray[np.uint8]:
        """GPU-accelerated 2D convolution filter.

        Args:
            frame: Input image
            ddepth: Desired depth of output image
            kernel: Convolution kernel

        Returns:
            Filtered image
        """
        if not self._opencl_available:
            return cv2.filter2D(frame, ddepth, kernel)

        try:
            # Upload to GPU
            umat = cv2.UMat(frame)
            kernel_umat = cv2.UMat(kernel)
            # Filter on GPU
            filtered = cv2.filter2D(umat, ddepth, kernel_umat)
            # Download from GPU
            return filtered.get()
        except Exception as e:
            logger.warning(f"GPU filter2D failed, falling back to CPU: {e}")
            return cv2.filter2D(frame, ddepth, kernel)

    def warp_perspective(
        self,
        frame: NDArray[np.uint8],
        matrix: NDArray[np.float64],
        size: tuple[int, int],
        flags: int = cv2.INTER_LINEAR,
    ) -> NDArray[np.uint8]:
        """GPU-accelerated perspective transformation.

        Args:
            frame: Input image
            matrix: 3x3 transformation matrix
            size: Output image size (width, height)
            flags: Interpolation flags

        Returns:
            Transformed image
        """
        if not self._opencl_available:
            return cv2.warpPerspective(frame, matrix, size, flags=flags)

        try:
            # Upload to GPU
            umat = cv2.UMat(frame)
            # Transform on GPU
            warped = cv2.warpPerspective(umat, matrix, size, flags=flags)
            # Download from GPU
            return warped.get()
        except Exception as e:
            logger.warning(f"GPU warp perspective failed, falling back to CPU: {e}")
            return cv2.warpPerspective(frame, matrix, size, flags=flags)

    def get_info(self) -> dict:
        """Get GPU acceleration information.

        Returns:
            Dictionary with GPU info
        """
        info = {
            "gpu_available": self._gpu_available,
            "opencl_available": self._opencl_available,
            "vaapi_available": self._vaapi_available,
            "initialized": self._initialized,
        }

        # DO NOT query OpenCL device info if not available - can cause hangs
        if self._opencl_available:
            logger.warning("OpenCL device info query skipped to prevent potential hang")
            info["opencl_note"] = (
                "Device info not available (disabled to prevent hangs)"
            )

        if self._vaapi_available:
            info["vaapi_driver"] = os.environ.get("LIBVA_DRIVER_NAME", "unknown")

        return info


# Global singleton instance
_gpu_accelerator: Optional[GPUAccelerator] = None


def get_gpu_accelerator() -> GPUAccelerator:
    """Get the global GPU accelerator instance.

    Returns:
        GPUAccelerator singleton instance
    """
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator()
    return _gpu_accelerator


def configure_vaapi_env():
    """Configure environment variables for VAAPI hardware acceleration.

    This should be called early in the application startup before any
    video capture or processing begins.
    """
    # Set VAAPI driver (iHD for Intel GPU)
    if "LIBVA_DRIVER_NAME" not in os.environ:
        os.environ["LIBVA_DRIVER_NAME"] = "iHD"
        logger.info("Set LIBVA_DRIVER_NAME=iHD for Intel VAAPI")

    # Set FFmpeg hardware acceleration
    if "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
        # Enable hardware acceleration for video decoding
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "hwaccel;vaapi|hwaccel_device;/dev/dri/renderD128"
        )
        logger.info("Set OPENCV_FFMPEG_CAPTURE_OPTIONS for hardware video decoding")

    # Additional VAAPI options for better performance
    if "LIBVA_MESSAGING_LEVEL" not in os.environ:
        os.environ["LIBVA_MESSAGING_LEVEL"] = "1"  # Reduce verbose logging

    logger.info("VAAPI environment configured for GPU acceleration")
