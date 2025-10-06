#!/usr/bin/env python3
"""Simple video test without threading to debug the issue."""

import cv2
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing video file reading...")

# Test basic OpenCV
video_path = "demo.mkv"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"ERROR: Could not open {video_path}")
    sys.exit(1)

print(f"Video opened successfully")
print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Frame count: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

# Read a frame
ret, frame = cap.read()
if not ret:
    print("ERROR: Could not read frame")
    sys.exit(1)

print(f"Frame read successfully: {frame.shape}")

# Test detection
print("\nTesting ball detection...")
import importlib.util
import types

# Create kinect2 stub
kinect2_module = types.ModuleType('backend.vision.kinect2_capture')
kinect2_module.KINECT2_AVAILABLE = False
sys.modules['backend.vision.kinect2_capture'] = kinect2_module

# Import models
models_path = Path('.') / 'backend' / 'vision' / 'models.py'
spec = importlib.util.spec_from_file_location('backend.vision.models', models_path)
models_module = importlib.util.module_from_spec(spec)
sys.modules['backend.vision.models'] = models_module
spec.loader.exec_module(models_module)

# Import ball detector
balls_path = Path('.') / 'backend' / 'vision' / 'detection' / 'balls.py'
spec = importlib.util.spec_from_file_location('backend.vision.detection.balls', balls_path)
balls_module = importlib.util.module_from_spec(spec)
sys.modules['backend.vision.detection.balls'] = balls_module
spec.loader.exec_module(balls_module)

BallDetector = balls_module.BallDetector

detector_config = {
    'detection_method': 'combined',
    'min_radius': 10,
    'max_radius': 40,
    'expected_radius': 20,
    'debug_mode': False,
}
detector = BallDetector(detector_config)

print("Ball detector initialized")

# Detect balls in frame
balls = detector.detect_balls(frame)
print(f"Detected {len(balls)} balls")

# Show frame with detections
for i, ball in enumerate(balls):
    x, y = ball.position
    r = ball.radius
    cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
    cv2.putText(frame, f"Ball {i}", (int(x)-20, int(y)-int(r)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display
cv2.imshow("Video Test", frame)
print("\nPress any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()

cap.release()
print("Done!")
