#!/usr/bin/env python3
"""Test script to verify ball detection and fisheye correction are working."""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add backend to path
backend_path = Path(__file__).parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

import logging

from vision.calibration.camera import CameraCalibrator
from vision.capture import CameraCapture
from vision.detection.balls import BallDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test ball detection with fisheye correction."""
    logger.info("Starting detection test...")

    # Initialize camera
    camera_config = {
        "device_id": 0,
        "backend": "auto",
        "resolution": (1920, 1080),
        "fps": 30,
        "buffer_size": 1,
    }

    camera = CameraCapture(camera_config)
    if not camera.start_capture():
        logger.error("Failed to start camera")
        return

    logger.info("Camera started successfully")

    # Check for fisheye calibration
    calibration_path = backend_path / "calibration/camera_fisheye_default.yaml"
    calibrator = None
    if calibration_path.exists():
        logger.info(f"Loading fisheye calibration from {calibration_path}")
        calibrator = CameraCalibrator()
        if calibrator.load_calibration(str(calibration_path)):
            logger.info("✅ Fisheye calibration loaded successfully")
        else:
            logger.warning("❌ Failed to load fisheye calibration")
            calibrator = None
    else:
        logger.warning(f"❌ No fisheye calibration found at {calibration_path}")

    # Initialize ball detector
    logger.info("Initializing ball detector...")
    ball_detector = BallDetector()
    logger.info("Ball detector initialized")

    # Capture a frame
    logger.info("Capturing frame...")
    import time

    time.sleep(1)  # Give camera time to stabilize

    frame_data = camera.get_latest_frame()
    if frame_data is None:
        logger.error("Failed to capture frame")
        camera.stop_capture()
        return

    frame, frame_info = frame_data
    logger.info(f"Frame captured: {frame.shape}")

    # Save raw frame
    raw_path = backend_path / "test_results/01_raw_frame.jpg"
    raw_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(raw_path), frame)
    logger.info(f"✅ Raw frame saved to {raw_path}")

    # Apply fisheye correction if available
    corrected_frame = frame
    if calibrator:
        logger.info("Applying fisheye correction...")
        corrected_frame = calibrator.undistort(frame)
        corrected_path = backend_path / "test_results/02_fisheye_corrected.jpg"
        cv2.imwrite(str(corrected_path), corrected_frame)
        logger.info(f"✅ Fisheye-corrected frame saved to {corrected_path}")
    else:
        logger.info("⚠️  Skipping fisheye correction (no calibration)")

    # Run ball detection
    logger.info("Running ball detection...")
    detected_balls = ball_detector.detect(corrected_frame)
    logger.info(f"✅ Detected {len(detected_balls)} balls")

    # Draw detections on frame
    annotated_frame = corrected_frame.copy()
    for ball in detected_balls:
        # Draw circle for ball
        center = (int(ball.position[0]), int(ball.position[1]))
        radius = int(ball.radius)

        # Different colors for different ball types
        if ball.ball_type.name == "CUE":
            color = (255, 255, 255)  # White
        elif ball.ball_type.name.startswith("SOLID"):
            color = (0, 255, 0)  # Green
        elif ball.ball_type.name.startswith("STRIPE"):
            color = (0, 165, 255)  # Orange
        elif ball.ball_type.name == "EIGHT":
            color = (0, 0, 0)  # Black
        else:
            color = (128, 128, 128)  # Gray

        # Draw filled circle
        cv2.circle(annotated_frame, center, radius, color, -1)

        # Draw outline
        cv2.circle(annotated_frame, center, radius, (0, 0, 255), 2)

        # Draw ball ID
        text = f"{ball.number}" if ball.number else ball.ball_type.name[:3]
        cv2.putText(
            annotated_frame,
            text,
            (center[0] - 10, center[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        logger.info(
            f"  Ball {ball.number}: type={ball.ball_type.name}, "
            f"pos={ball.position}, radius={ball.radius:.1f}, "
            f"confidence={ball.confidence:.2f}"
        )

    # Save annotated frame
    annotated_path = backend_path / "test_results/03_detected_balls.jpg"
    cv2.imwrite(str(annotated_path), annotated_frame)
    logger.info(f"✅ Annotated frame saved to {annotated_path}")

    # Cleanup
    camera.stop_capture()
    logger.info("Camera stopped")

    # Summary
    print("\n" + "=" * 60)
    print("DETECTION TEST SUMMARY")
    print("=" * 60)
    print(f"Fisheye Correction: {'✅ ENABLED' if calibrator else '❌ DISABLED'}")
    print(f"Balls Detected: {len(detected_balls)}")
    print(f"\nResults saved to: {backend_path}/test_results/")
    print("  - 01_raw_frame.jpg (raw camera frame)")
    if calibrator:
        print("  - 02_fisheye_corrected.jpg (after correction)")
    print("  - 03_detected_balls.jpg (with ball annotations)")
    print("=" * 60)


if __name__ == "__main__":
    main()
