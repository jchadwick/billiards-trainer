"""Enhanced Camera Module with Fisheye Correction and Preprocessing.

This module combines camera capture, fisheye correction, and image preprocessing
into a single efficient pipeline for both vision processing and web streaming.
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional, Any

import cv2
import numpy as np

from ..vision.calibration.camera import CameraCalibrator
from ..vision.preprocessing import ImagePreprocessor


@dataclass
class EnhancedCameraConfig:
    """Configuration for enhanced camera module."""

    # Camera settings
    device_id: int = 0
    resolution: tuple[int, int] = (1920, 1080)
    fps: int = 30

    # Fisheye correction
    enable_fisheye_correction: bool = True
    calibration_file: Optional[str] = "calibration/camera_fisheye.yaml"

    # Preprocessing
    enable_preprocessing: bool = True
    brightness: float = 0.0  # -100 to 100
    contrast: float = 1.0    # 0.5 to 3.0
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    # Performance
    enable_gpu: bool = False
    buffer_size: int = 1


class EnhancedCameraModule:
    """Camera module with integrated fisheye correction and preprocessing."""

    def __init__(self, config: EnhancedCameraConfig):
        self.config = config
        self.capture = None
        self.running = False

        # Threading components
        self._lock = threading.Lock()
        self._current_frame = None
        self._processed_frame = None
        self._capture_thread = None

        # Initialize calibration
        self.calibrator = None
        self.undistort_map1 = None
        self.undistort_map2 = None
        if config.enable_fisheye_correction:
            self._load_calibration()

        # Initialize preprocessor
        self.preprocessor = None
        if config.enable_preprocessing:
            self._init_preprocessor()

    def _load_calibration(self):
        """Load fisheye calibration data."""
        if self.config.calibration_file:
            try:
                # Load calibration file
                fs = cv2.FileStorage(self.config.calibration_file, cv2.FILE_STORAGE_READ)

                camera_matrix = fs.getNode("camera_matrix").mat()
                dist_coeffs = fs.getNode("dist_coeffs").mat()

                # Pre-compute undistortion maps for efficiency
                h, w = self.config.resolution[1], self.config.resolution[0]
                new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    camera_matrix, dist_coeffs, (w, h), np.eye(3)
                )

                self.undistort_map1, self.undistort_map2 = cv2.fisheye.initUndistortRectifyMap(
                    camera_matrix, dist_coeffs, np.eye(3),
                    new_camera_matrix, (w, h), cv2.CV_16SC2
                )

                print(f"Loaded fisheye calibration from {self.config.calibration_file}")
            except Exception as e:
                print(f"Failed to load calibration: {e}")
                print("Fisheye correction disabled")
                self.config.enable_fisheye_correction = False

    def _init_preprocessor(self):
        """Initialize image preprocessor."""
        self.clahe = None
        if self.config.enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size)
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

        # Configure camera
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
        self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

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

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply fisheye correction and preprocessing pipeline."""
        # Step 1: Fisheye correction
        if self.config.enable_fisheye_correction and self.undistort_map1 is not None:
            frame = cv2.remap(
                frame, self.undistort_map1, self.undistort_map2,
                interpolation=cv2.INTER_LINEAR
            )

        # Step 2: Image preprocessing
        if self.config.enable_preprocessing:
            # Brightness and contrast adjustment
            if self.config.brightness != 0 or self.config.contrast != 1.0:
                frame = cv2.convertScaleAbs(
                    frame,
                    alpha=self.config.contrast,
                    beta=self.config.brightness
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

            # Bilateral filter for noise reduction while preserving edges
            # (Important for ball detection)
            frame = cv2.bilateralFilter(frame, d=5, sigmaColor=50, sigmaSpace=50)

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

    def get_frame_for_streaming(self,
                                quality: int = 80,
                                max_width: Optional[int] = None,
                                max_height: Optional[int] = None) -> Optional[bytes]:
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

    def adjust_preprocessing(self,
                           brightness: Optional[float] = None,
                           contrast: Optional[float] = None,
                           clahe_enabled: Optional[bool] = None):
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
            "fps": self.config.fps
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
        enable_clahe=True
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
