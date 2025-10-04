"""Enhanced Camera Module with Fisheye Correction and Preprocessing.

This module combines camera capture, fisheye correction, and image preprocessing
into a single efficient pipeline for both vision processing and web streaming.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from ..vision.calibration.camera import CameraCalibrator
from ..vision.preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)


@dataclass
class EnhancedCameraConfig:
    """Configuration for enhanced camera module."""

    # Camera settings
    device_id: int = 0
    resolution: Optional[tuple[int, int]] = None  # Auto-detect if None
    fps: int = 30

    # Fisheye correction
    enable_fisheye_correction: bool = True
    calibration_file: Optional[str] = "calibration/camera_fisheye.yaml"

    # Table cropping
    enable_table_crop: bool = True

    # Preprocessing
    enable_preprocessing: bool = True
    brightness: float = 0.0  # -100 to 100
    contrast: float = 1.0  # 0.5 to 3.0
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    # Performance
    enable_gpu: bool = False
    buffer_size: int = 1


class EnhancedCameraModule:
    """Camera module with integrated fisheye correction and preprocessing."""

    def __init__(
        self,
        config: EnhancedCameraConfig,
        event_loop=None,
        frame_callback=None,
    ):
        """Initialize enhanced camera module.

        Args:
            config: Camera configuration settings
            event_loop: Optional asyncio event loop for WebSocket callbacks
            frame_callback: Optional async callback for frame broadcasting
        """
        self.config = config
        self.capture = None
        self.running = False

        # Threading components
        self._lock = threading.Lock()
        self._current_frame = None
        self._processed_frame = None
        self._capture_thread = None

        # Async integration for WebSocket broadcasting
        self._event_loop = event_loop
        self._frame_callback = frame_callback

        # Table crop region (cached after first detection)
        self._table_crop_region = None

        # Detect actual camera resolution BEFORE initializing calibration
        self._detect_actual_resolution()

        # Initialize calibration (uses the verified resolution from config)
        self.calibrator = None
        self.undistort_map1 = None
        self.undistort_map2 = None
        if config.enable_fisheye_correction:
            self._load_calibration()

        # Initialize preprocessor
        self.preprocessor = None
        if config.enable_preprocessing:
            self._init_preprocessor()

    def _detect_actual_resolution(self):
        """Detect the actual resolution supported by the camera.

        Opens the camera temporarily to verify what resolution it actually provides,
        then updates self.config.resolution to match. This ensures undistortion maps
        and all processing are created for the correct resolution.
        """
        temp_cap = cv2.VideoCapture(self.config.device_id)
        if not temp_cap.isOpened():
            logger.warning(
                f"Could not open camera {self.config.device_id} to detect resolution"
            )
            return

        try:
            # If resolution was specified, try to set it
            if self.config.resolution is not None:
                temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
                temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])

            # Get actual resolution (either what we set or camera's default)
            actual_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self.config.resolution is None:
                # Auto-detected camera's native resolution
                print(
                    f"EnhancedCameraModule: Auto-detected camera resolution: {actual_width}x{actual_height}"
                )
                self.config.resolution = (actual_width, actual_height)
            elif (
                actual_width != self.config.resolution[0]
                or actual_height != self.config.resolution[1]
            ):
                print(
                    f"EnhancedCameraModule: Requested {self.config.resolution[0]}x{self.config.resolution[1]}, but camera only supports {actual_width}x{actual_height}"
                )
                print(
                    "EnhancedCameraModule: Updating config to use actual camera resolution"
                )
                self.config.resolution = (actual_width, actual_height)
            else:
                print(
                    f"EnhancedCameraModule: Camera resolution verified: {actual_width}x{actual_height}"
                )
        finally:
            temp_cap.release()

    def _load_calibration(self):
        """Load fisheye calibration data."""
        if self.config.calibration_file:
            fs = None
            try:
                # Load calibration file
                fs = cv2.FileStorage(
                    self.config.calibration_file, cv2.FILE_STORAGE_READ
                )

                # Check if file was opened successfully
                if not fs.isOpened():
                    raise RuntimeError(
                        f"Could not open calibration file: {self.config.calibration_file}"
                    )

                # Get nodes and check they exist before calling mat()
                camera_matrix_node = fs.getNode("camera_matrix")
                dist_coeffs_node = fs.getNode("dist_coeffs")

                if camera_matrix_node.empty() or dist_coeffs_node.empty():
                    raise RuntimeError(
                        "Calibration file is missing required data (camera_matrix or dist_coeffs)"
                    )

                # Read the matrices - need to do this before releasing FileStorage
                # but wrap in try/except to catch OpenCV errors
                try:
                    camera_matrix = camera_matrix_node.mat()
                    dist_coeffs = dist_coeffs_node.mat()
                except Exception as mat_error:
                    raise RuntimeError(
                        f"Failed to read calibration matrices: {mat_error}"
                    )

                # Release FileStorage after reading the data
                fs.release()
                fs = None

                # Validate the data
                if camera_matrix is None or dist_coeffs is None:
                    raise RuntimeError("Calibration matrices are None")
                if camera_matrix.shape != (3, 3):
                    raise RuntimeError(
                        f"Invalid camera matrix shape: {camera_matrix.shape}, expected (3, 3)"
                    )
                if dist_coeffs.size < 4:
                    raise RuntimeError(
                        f"Invalid distortion coefficients shape: {dist_coeffs.shape}, expected at least 4 elements"
                    )

                # Pre-compute undistortion maps for efficiency
                h, w = self.config.resolution[1], self.config.resolution[0]

                # Use standard camera model (not fisheye) to match calibration
                # The calibration uses cv2.calibrateCamera, so we must use standard undistortion
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                    camera_matrix,
                    dist_coeffs,
                    (w, h),
                    1,  # alpha=1 keeps all pixels
                    (w, h),
                )

                self.undistort_map1, self.undistort_map2 = cv2.initUndistortRectifyMap(
                    camera_matrix,
                    dist_coeffs,
                    None,  # No rectification
                    new_camera_matrix,
                    (w, h),
                    cv2.CV_16SC2,
                )

                print(f"Loaded fisheye calibration from {self.config.calibration_file}")
            except Exception as e:
                print(f"Failed to load calibration: {e}")
                print("Fisheye correction disabled")
                self.config.enable_fisheye_correction = False
            finally:
                # Ensure FileStorage is always released
                if fs is not None:
                    fs.release()

    def _init_preprocessor(self):
        """Initialize image preprocessor."""
        self.clahe = None
        if self.config.enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size),
            )

    def start_capture(self) -> bool:
        """Start camera capture with processing pipeline."""
        if self.running:
            return True

        # Start capture thread
        self.running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        # Wait for first frame
        for _ in range(50):  # 5 second timeout
            if self._current_frame is not None:
                return True
            time.sleep(0.1)

        return False

    def stop_capture(self):
        """Stop camera capture."""
        self.running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self.capture:
            self.capture.release()

    def _capture_loop(self):
        """Main capture and processing loop."""
        # Open camera
        self.capture = cv2.VideoCapture(self.config.device_id)

        if not self.capture.isOpened():
            print(f"Failed to open camera {self.config.device_id}")
            return

        # Configure camera (using already-verified resolution from __init__)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
        self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

        # Configure camera for better color reproduction
        # Enable auto white balance to let camera handle it
        self.capture.set(cv2.CAP_PROP_AUTO_WB, 1)  # Enable auto white balance

        # Try AUTO exposure with compensation
        self.capture.set(
            cv2.CAP_PROP_AUTO_EXPOSURE, 0.25
        )  # Auto mode (0.75 = on, 0.25 = off for some cameras)
        self.capture.set(cv2.CAP_PROP_EXPOSURE, -13)  # Very low exposure value

        # Camera controls - try absolute minimum values
        self.capture.set(cv2.CAP_PROP_BRIGHTNESS, 0)  # Minimal brightness
        self.capture.set(cv2.CAP_PROP_CONTRAST, 32)  # Default contrast
        self.capture.set(cv2.CAP_PROP_SATURATION, 64)  # Default saturation
        self.capture.set(cv2.CAP_PROP_GAIN, 0)  # No gain
        self.capture.set(cv2.CAP_PROP_BACKLIGHT, 0)  # Disable backlight compensation

        logger.info(
            f"Camera settings - Exposure: {self.capture.get(cv2.CAP_PROP_EXPOSURE)}, "
            f"Brightness: {self.capture.get(cv2.CAP_PROP_BRIGHTNESS)}, "
            f"Contrast: {self.capture.get(cv2.CAP_PROP_CONTRAST)}, "
            f"Saturation: {self.capture.get(cv2.CAP_PROP_SATURATION)}, "
            f"WB: {self.capture.get(cv2.CAP_PROP_WB_TEMPERATURE)}"
        )

        while self.running:
            ret, frame = self.capture.read()

            if not ret:
                continue

            # Apply processing pipeline
            processed = self._process_frame(frame)

            # Update shared buffers (thread-safe)
            with self._lock:
                self._current_frame = frame.copy()
                self._processed_frame = processed.copy()

            # Trigger async callback for WebSocket broadcasting (if configured)
            if self._event_loop and self._frame_callback:
                try:
                    # Schedule coroutine on the event loop from this sync thread
                    asyncio.run_coroutine_threadsafe(
                        self._frame_callback(
                            processed.copy(),
                            processed.shape[1],  # width
                            processed.shape[0],  # height
                        ),
                        self._event_loop,
                    )
                except Exception as e:
                    # Don't let callback errors crash the capture loop
                    print(f"Error in frame callback: {e}")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply fisheye correction and preprocessing pipeline."""
        # Step 1: Fisheye correction
        if self.config.enable_fisheye_correction and self.undistort_map1 is not None:
            frame = cv2.remap(
                frame,
                self.undistort_map1,
                self.undistort_map2,
                interpolation=cv2.INTER_LINEAR,
            )

        # Step 1.5: Aggressive software correction for overexposed camera
        # The camera hardware auto-exposure cannot be disabled via OpenCV
        # So we need to fix it in software

        # First, drastically reduce brightness/exposure in BGR space
        frame = cv2.convertScaleAbs(frame, alpha=0.4, beta=-50)

        # Then boost saturation and reduce value in HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)

        # Boost saturation significantly
        s = np.clip(s * 2.0, 0, 255)

        # Reduce brightness further
        v = np.clip(v * 0.7, 0, 255)

        hsv = cv2.merge([h, s, v]).astype(np.uint8)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Step 2: Table cropping (after fisheye correction)
        if self.config.enable_table_crop:
            frame = self._crop_to_table(frame)

        # Step 3: Image preprocessing
        if self.config.enable_preprocessing:
            # Brightness and contrast adjustment
            if self.config.brightness != 0 or self.config.contrast != 1.0:
                frame = cv2.convertScaleAbs(
                    frame, alpha=self.config.contrast, beta=self.config.brightness
                )

            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if self.clahe is not None:
                # Convert to LAB color space
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)

                # Apply CLAHE to L channel
                l = self.clahe.apply(l)

                # Merge and convert back
                lab = cv2.merge([l, a, b])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Bilateral filter disabled - too computationally expensive for real-time streaming
            # For vision processing, apply filters separately on demand
            # frame = cv2.bilateralFilter(frame, d=5, sigmaColor=50, sigmaSpace=50)

        return frame

    def _crop_to_table(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame to table boundaries using felt detection.

        Detects the green felt surface and crops to its bounding box.
        Caches the crop region for performance.
        """
        # Use cached crop region if available
        if self._table_crop_region is not None:
            x, y, w, h = self._table_crop_region
            return frame[y : y + h, x : x + w]

        # Detect table region (only once)
        try:
            # Convert to HSV and detect green felt
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find largest contour (the felt surface)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Add small padding (5% on each side)
                padding_x = int(w * 0.05)
                padding_y = int(h * 0.05)

                x = max(0, x - padding_x)
                y = max(0, y - padding_y)
                w = min(frame.shape[1] - x, w + 2 * padding_x)
                h = min(frame.shape[0] - y, h + 2 * padding_y)

                # Cache the crop region
                self._table_crop_region = (x, y, w, h)

                return frame[y : y + h, x : x + w]
        except Exception as e:
            print(f"Table crop failed: {e}")

        # Return original frame if detection fails
        return frame

    def get_frame(self, processed: bool = True) -> Optional[np.ndarray]:
        """Get current frame for vision processing.

        Args:
            processed: If True, return processed frame. If False, return raw frame.

        Returns:
            Current frame or None if not available.
        """
        with self._lock:
            if processed and self._processed_frame is not None:
                return self._processed_frame.copy()
            elif self._current_frame is not None:
                return self._current_frame.copy()
        return None

    def get_frame_for_streaming(
        self,
        quality: int = 80,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
    ) -> Optional[bytes]:
        """Get JPEG-encoded frame for streaming.

        Args:
            quality: JPEG quality (1-100)
            max_width: Maximum width for resizing
            max_height: Maximum height for resizing

        Returns:
            JPEG bytes or None if not available.
        """
        frame = self.get_frame(processed=True)

        if frame is None:
            return None

        # Resize if requested
        if max_width or max_height:
            h, w = frame.shape[:2]
            scale = 1.0

            if max_width:
                scale = min(scale, max_width / w)
            if max_height:
                scale = min(scale, max_height / h)

            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Encode to JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)

        if success:
            return buffer.tobytes()

        return None

    def calibrate_fisheye(self, calibration_images: list) -> bool:
        """Calibrate fisheye lens using calibration images.

        Args:
            calibration_images: List of calibration pattern images

        Returns:
            True if calibration successful
        """
        # Use existing CameraCalibrator class
        calibrator = CameraCalibrator()

        # Detect calibration pattern in images
        for img in calibration_images:
            calibrator.add_calibration_image(img)

        # Perform calibration
        if calibrator.calibrate():
            # Save calibration
            calibrator.save_calibration(self.config.calibration_file)

            # Reload calibration maps
            self._load_calibration()

            return True

        return False

    def adjust_preprocessing(
        self,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        clahe_enabled: Optional[bool] = None,
    ):
        """Dynamically adjust preprocessing parameters.

        Args:
            brightness: New brightness value (-100 to 100)
            contrast: New contrast value (0.5 to 3.0)
            clahe_enabled: Enable/disable CLAHE
        """
        if brightness is not None:
            self.config.brightness = np.clip(brightness, -100, 100)

        if contrast is not None:
            self.config.contrast = np.clip(contrast, 0.5, 3.0)

        if clahe_enabled is not None:
            self.config.enable_clahe = clahe_enabled
            if clahe_enabled and self.clahe is None:
                self._init_preprocessor()

    def get_statistics(self) -> dict[str, Any]:
        """Get module statistics."""
        return {
            "running": self.running,
            "fisheye_correction_enabled": self.config.enable_fisheye_correction,
            "preprocessing_enabled": self.config.enable_preprocessing,
            "brightness": self.config.brightness,
            "contrast": self.config.contrast,
            "clahe_enabled": self.config.enable_clahe,
            "resolution": self.config.resolution,
            "fps": self.config.fps,
        }


# Example usage for single-application architecture
if __name__ == "__main__":
    config = EnhancedCameraConfig(
        device_id=0,
        resolution=(1920, 1080),
        fps=30,
        enable_fisheye_correction=True,
        calibration_file="calibration/camera.yaml",
        enable_preprocessing=True,
        brightness=10,
        contrast=1.2,
        enable_clahe=True,
    )

    camera = EnhancedCameraModule(config)

    if camera.start_capture():
        print("Camera started successfully")

        # For vision processing - get processed frames
        vision_frame = camera.get_frame(processed=True)

        # For web streaming - get JPEG bytes
        stream_data = camera.get_frame_for_streaming(quality=80, max_width=1280)

        print(f"Statistics: {camera.get_statistics()}")

    camera.stop_capture()
