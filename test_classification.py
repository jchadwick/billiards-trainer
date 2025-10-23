#!/usr/bin/env python3
"""Test ball classification with debug output."""

import cv2
import logging
import sys
from backend.vision.detection.balls import BallDetector
from config import config

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Set all vision loggers to DEBUG
for logger_name in ['backend.vision', 'backend.vision.detection', 'backend.vision.detection.balls']:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

print("Loading video frame...")
cap = cv2.VideoCapture("assets/demo3.mp4")
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to load frame!")
    sys.exit(1)

print(f"Frame loaded: {frame.shape}")

print("\nInitializing ball detector...")
detector = BallDetector(config.get("vision.ball_detection", {}))

print("\nDetecting balls...")
balls = detector.detect_balls(frame)

print(f"\n{'='*80}")
print(f"RESULTS: Found {len(balls)} balls")
print(f"{'='*80}\n")

for i, ball in enumerate(balls, 1):
    print(f"Ball {i}:")
    print(f"  Type: {ball.ball_type.name}")
    print(f"  Number: {ball.number}")
    print(f"  Position: ({ball.position[0]:.1f}, {ball.position[1]:.1f})")
    print(f"  Confidence: {ball.confidence:.3f}")
    print()

# Count duplicates
from collections import Counter
types = [ball.ball_type.name for ball in balls]
numbers = [ball.number for ball in balls if ball.number is not None]

print(f"\nType distribution:")
for ball_type, count in Counter(types).items():
    marker = " ⚠️  DUPLICATE!" if count > 1 else ""
    print(f"  {ball_type}: {count}{marker}")

if numbers:
    print(f"\nNumber distribution:")
    for number, count in Counter(numbers).items():
        marker = " ⚠️  DUPLICATE!" if count > 1 else ""
        print(f"  {number}: {count}{marker}")
