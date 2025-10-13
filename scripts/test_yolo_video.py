#!/usr/bin/env python3
"""Quick test to verify YOLO detection works in video_debugger context."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

import cv2
from backend.vision.detection.yolo_detector import YOLODetector

def test_yolo_detection():
    """Test YOLO detection on a single frame."""

    # Initialize YOLO detector
    print("Initializing YOLO detector...")
    detector = YOLODetector(
        model_path="models/yolov8n-billiards.onnx",
        device="cpu",
        confidence=0.4,
        nms_threshold=0.45,
    )

    # Load a test frame from video
    print("Loading test video frame...")
    cap = cv2.VideoCapture("demo.mkv")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read frame from video")
        return False

    print(f"Frame shape: {frame.shape}")

    # Detect balls
    print("Running ball detection...")
    detections = detector.detect_balls(frame)

    print(f"\nDetection results:")
    print(f"  Total detections: {len(detections)}")

    for i, det in enumerate(detections):
        print(f"  Detection {i+1}:")
        print(f"    Class ID: {det.class_id}")
        print(f"    Class Name: {det.class_name}")
        print(f"    Confidence: {det.confidence:.3f}")
        print(f"    BBox: {det.bbox}")
        print(f"    Center: {det.center}")

    # Get statistics
    stats = detector.get_statistics()
    print(f"\nDetector statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return len(detections) > 0

if __name__ == "__main__":
    success = test_yolo_detection()
    if success:
        print("\n✓ YOLO detection test PASSED")
        sys.exit(0)
    else:
        print("\n✗ YOLO detection test FAILED")
        sys.exit(1)
