#!/usr/bin/env python3
"""Test hybrid YOLO detection + OpenCV classification."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
from backend.vision.detection.yolo_detector import YOLODetector
from backend.vision.detection.balls import BallDetector
from backend.vision.detection.detector_adapter import yolo_detections_to_balls

def test_hybrid_detection():
    """Test YOLO detection with OpenCV classification."""

    print("=== Hybrid Detection Test ===\n")

    # Initialize YOLO detector
    print("1. Initializing YOLO detector...")
    yolo_detector = YOLODetector(
        model_path="models/yolov8n-billiards.onnx",
        device="cpu",
        confidence=0.4,
    )

    # Initialize OpenCV classifier
    print("2. Initializing OpenCV classifier...")
    opencv_classifier = BallDetector({"detection_method": "combined", "debug_mode": False})

    # Load test frame
    print("3. Loading test frame...")
    cap = cv2.VideoCapture("demo.mkv")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read frame")
        return False

    # YOLO detection
    print("4. Running YOLO detection...")
    yolo_detections = yolo_detector.detect_balls(frame)
    print(f"   YOLO detected: {len(yolo_detections)} balls")

    # Convert and classify
    print("5. Converting detections and classifying with OpenCV...")
    classified_balls = []

    for i, det in enumerate(yolo_detections):
        # Convert to Ball object
        detection_dict = {
            "bbox": det.bbox,
            "confidence": det.confidence,
            "class_id": det.class_id,
            "class_name": det.class_name,
        }
        balls = yolo_detections_to_balls(
            [detection_dict],
            (frame.shape[0], frame.shape[1]),
            min_confidence=0.25,
            bbox_format="xyxy",
        )

        for ball in balls:
            if det.class_name == "ball":
                # Extract region
                x, y = ball.position
                r = ball.radius
                x1, y1 = max(0, int(x - r * 1.2)), max(0, int(y - r * 1.2))
                x2, y2 = min(frame.shape[1], int(x + r * 1.2)), min(frame.shape[0], int(y + r * 1.2))
                ball_region = frame[y1:y2, x1:x2]

                if ball_region.size > 0:
                    # Classify with OpenCV
                    ball_type, conf, ball_number = opencv_classifier.classify_ball_type(
                        ball_region, ball.position, r
                    )
                    ball.ball_type = ball_type
                    ball.number = ball_number

                    print(f"   Ball {i+1}: pos=({x:.0f},{y:.0f}), "
                          f"type={ball_type}, num={ball_number}, "
                          f"YOLO_conf={det.confidence:.3f}, class_conf={conf:.3f}")

            classified_balls.append(ball)

    print(f"\n6. Final results: {len(classified_balls)} balls classified")

    # Count by type
    from collections import Counter
    types = Counter(str(b.ball_type) for b in classified_balls)
    print(f"   Ball types: {dict(types)}")

    return True

if __name__ == "__main__":
    success = test_hybrid_detection()
    sys.exit(0 if success else 1)
