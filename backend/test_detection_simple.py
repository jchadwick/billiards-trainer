#!/usr/bin/env python3
"""Simple test script to verify ball detection and fisheye correction without CameraCapture."""

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
from vision.detection.balls import BallDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test ball detection with fisheye correction."""
    logger.info("Starting simple detection test...")

    # Initialize camera directly with OpenCV
    logger.info("Opening camera device 0...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Failed to open camera")
        return

    logger.info("Camera opened successfully")

    # Set camera parameters
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Get actual resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera resolution: {width}x{height}")

    # Skip fisheye calibration for now (would need proper format)
    calibrator = None
    logger.info("⚠️  Skipping fisheye correction for this test")

    # Initialize ball detector with minimal config
    logger.info("Initializing ball detector...")
    ball_detector_config = {
        "detection_method": "hough",
        "min_radius": 10,
        "max_radius": 50,
        "min_confidence": 0.5,
    }
    ball_detector = BallDetector(ball_detector_config)
    logger.info("Ball detector initialized")

    # Capture a frame
    logger.info("Capturing frame...")
    import time

    time.sleep(1)  # Give camera time to stabilize

    # Read several frames to flush buffer
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    if not ret or frame is None:
        logger.error("Failed to capture frame")
        cap.release()
        return

    logger.info(f"Frame captured: {frame.shape}")

    # Create output directory
    output_dir = backend_path / "test_results"
    output_dir.mkdir(exist_ok=True)

    # Save raw frame
    raw_path = output_dir / "01_raw_frame.jpg"
    cv2.imwrite(str(raw_path), frame)
    logger.info(f"✅ Raw frame saved to {raw_path}")

    # Apply fisheye correction if available
    corrected_frame = frame
    if calibrator:
        logger.info("Applying fisheye correction...")
        corrected_frame = calibrator.undistort(frame)
        corrected_path = output_dir / "02_fisheye_corrected.jpg"
        cv2.imwrite(str(corrected_path), corrected_frame)
        logger.info(f"✅ Fisheye-corrected frame saved to {corrected_path}")
    else:
        logger.info("⚠️  Skipping fisheye correction (no calibration)")

    # Run ball detection
    logger.info("Running ball detection...")
    detected_balls = ball_detector.detect_balls(corrected_frame)
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
    annotated_path = output_dir / "03_detected_balls.jpg"
    cv2.imwrite(str(annotated_path), annotated_frame)
    logger.info(f"✅ Annotated frame saved to {annotated_path}")

    # Cleanup
    cap.release()
    logger.info("Camera released")

    # Summary
    print("\n" + "=" * 60)
    print("DETECTION TEST SUMMARY")
    print("=" * 60)
    print(f"Fisheye Correction: {'✅ ENABLED' if calibrator else '❌ DISABLED'}")
    print(f"Balls Detected: {len(detected_balls)}")
    print(f"\nResults saved to: {output_dir}/")
    print("  - 01_raw_frame.jpg (raw camera frame)")
    if calibrator:
        print("  - 02_fisheye_corrected.jpg (after correction)")
    print("  - 03_detected_balls.jpg (with ball annotations)")
    print("=" * 60)


if __name__ == "__main__":
    main()
