#!/usr/bin/env python3
"""Minimal test script to diagnose vision module detection issues.

This script tests the VisionModule with a single frame from a video file
to determine why ball detections are not showing up in the video debugger.
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import backend components
from backend.vision import VisionModule


def test_vision_module(video_path: str):
    """Test VisionModule with a single frame from video file.

    Args:
        video_path: Path to video file to test
    """
    logger.info(f"=== Testing VisionModule with video: {video_path} ===")

    # Check video file exists
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return False

    # Open video file with OpenCV to extract a single frame
    logger.info("Opening video file with OpenCV...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error("Failed to open video file")
        return False

    # Read first frame
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        logger.error("Failed to read frame from video")
        return False

    logger.info(f"Successfully read frame: shape={frame.shape}, dtype={frame.dtype}")

    # Check config values BEFORE initializing VisionModule
    logger.info("\n=== Checking Config Values ===")
    try:
        from backend.config import config_manager
        yolo_model_path = config_manager.get("vision.detection.yolo_model_path")
        logger.info(f"Config yolo_model_path: {yolo_model_path}")

        import os
        if yolo_model_path:
            logger.info(f"Model path exists: {os.path.exists(yolo_model_path)}")
            logger.info(f"Model path absolute: {os.path.abspath(yolo_model_path)}")
        else:
            logger.error("yolo_model_path is None or empty!")
    except Exception as e:
        logger.error(f"Error checking config: {e}", exc_info=True)

    # Initialize VisionModule with same config as debugger
    logger.info("\n=== Initializing VisionModule ===")
    vision_config = {
        "camera_device_id": video_path,
        "enable_threading": False,
    }

    try:
        vision_module = VisionModule(vision_config)
        logger.info("VisionModule initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize VisionModule: {e}", exc_info=True)
        return False

    # Check detector state
    logger.info("\n=== Checking Detector State ===")
    if hasattr(vision_module, 'detector'):
        if vision_module.detector is None:
            logger.error("detector is None!")
        else:
            logger.info(f"detector type: {type(vision_module.detector)}")
            logger.info(f"detector.model_loaded: {vision_module.detector.model_loaded}")
            logger.info(f"detector.model_path: {vision_module.detector.model_path}")
            logger.info(f"detector.confidence: {vision_module.detector.confidence}")
            logger.info(f"detector.min_ball_size: {vision_module.detector.min_ball_size}")

            # Get model info
            model_info = vision_module.detector.get_model_info()
            logger.info(f"Model info: {model_info}")
    else:
        logger.error("VisionModule has no 'detector' attribute!")

    # Process single frame using internal method (same as debugger)
    logger.info("\n=== Processing Single Frame ===")
    try:
        detection_result = vision_module._process_single_frame(frame, 0, None)

        if detection_result is None:
            logger.error("detection_result is None!")
            return False

        logger.info(f"detection_result type: {type(detection_result)}")
        logger.info(f"detection_result.balls: {detection_result.balls}")
        logger.info(f"Number of balls detected: {len(detection_result.balls)}")

        if detection_result.balls:
            logger.info("\n=== Ball Detections ===")
            for i, ball in enumerate(detection_result.balls):
                logger.info(f"Ball {i}: position={ball.position}, radius={ball.radius}, "
                           f"ball_type={ball.ball_type}, confidence={getattr(ball, 'confidence', 'N/A')}")
        else:
            logger.warning("No balls detected!")

            # Try calling detector directly to see raw YOLO output
            logger.info("\n=== Testing Detector Directly ===")
            if vision_module.detector and vision_module.detector.model_loaded:
                logger.info("Calling detector.detect_balls() directly...")
                raw_detections = vision_module.detector.detect_balls(frame)
                logger.info(f"Raw YOLO detections: {len(raw_detections)}")

                if raw_detections:
                    for i, det in enumerate(raw_detections):
                        logger.info(f"Detection {i}: class_id={det.class_id}, "
                                   f"class_name={det.class_name}, confidence={det.confidence}, "
                                   f"bbox={det.bbox}, size={det.width}x{det.height}")
                else:
                    logger.warning("No raw YOLO detections returned!")

                # Try detect_balls_with_classification
                logger.info("\n=== Testing detect_balls_with_classification ===")
                classified_balls = vision_module.detector.detect_balls_with_classification(frame)
                logger.info(f"Classified balls: {len(classified_balls)}")

                if classified_balls:
                    for i, ball in enumerate(classified_balls):
                        logger.info(f"Classified ball {i}: position={ball.position}, "
                                   f"radius={ball.radius}, ball_type={ball.ball_type}")
                else:
                    logger.warning("No classified balls returned!")
            else:
                logger.error("Detector not loaded or not available!")

        # Check cue detection
        logger.info(f"\n=== Cue Detection ===")
        logger.info(f"detection_result.cue: {detection_result.cue}")

        # Check statistics
        logger.info(f"\n=== Detection Statistics ===")
        if hasattr(detection_result, 'statistics'):
            stats = detection_result.statistics
            logger.info(f"Frame number: {stats.frame_number}")
            logger.info(f"Processing time: {stats.processing_time:.2f}ms")
            logger.info(f"Balls detected: {stats.balls_detected}")
            logger.info(f"Balls tracked: {stats.balls_tracked}")
            logger.info(f"Cue detected: {stats.cue_detected}")
            logger.info(f"Detection confidence: {stats.detection_confidence:.3f}")

        return True

    except Exception as e:
        logger.error(f"Frame processing failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        # Default to demo2.mp4 if it exists
        video_path = str(Path(__file__).parent.parent / "demo2.mp4")
        if not Path(video_path).exists():
            logger.error("Please provide a video file path as argument")
            logger.error("Usage: python tools/test_vision_debug.py <video_file>")
            sys.exit(1)
    else:
        video_path = sys.argv[1]

    success = test_vision_module(video_path)

    if success:
        logger.info("\n=== TEST PASSED ===")
        sys.exit(0)
    else:
        logger.error("\n=== TEST FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
