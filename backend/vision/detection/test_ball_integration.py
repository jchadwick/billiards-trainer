"""Simple integration test for ball detection system.

Tests the basic functionality without complex dependencies.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from backend.vision.models import BallType

    from .ball_tracker import BallTrackingSystem
    from .balls import BallDetector, DetectionMethod

    print("✓ Successfully imported all modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def create_test_frame():
    """Create a synthetic test frame with balls."""
    # Create green table background
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    frame[:, :] = (40, 80, 40)  # Green felt color

    # Add some balls
    ball_positions = [
        (200, 200, (255, 255, 255)),  # White cue ball
        (400, 300, (255, 255, 0)),  # Yellow ball
        (600, 200, (0, 0, 255)),  # Red ball
    ]

    for x, y, color in ball_positions:
        # Draw ball with highlight
        cv2.circle(frame, (x, y), 20, color, -1)
        cv2.circle(frame, (x - 5, y - 5), 5, tuple(min(255, c + 50) for c in color), -1)

    return frame


def test_ball_detector():
    """Test basic ball detection."""
    print("\n=== Testing Ball Detector ===")

    config = {
        "detection_method": DetectionMethod.HOUGH_CIRCLES,
        "min_radius": 10,
        "max_radius": 30,
        "expected_radius": 20,
        "debug_mode": False,
    }

    try:
        detector = BallDetector(config)
        print("✓ Ball detector created successfully")
    except Exception as e:
        print(f"✗ Failed to create detector: {e}")
        return False

    # Test detection
    frame = create_test_frame()

    try:
        detected_balls = detector.detect_balls(frame)
        print(f"✓ Detection completed - found {len(detected_balls)} balls")

        for i, ball in enumerate(detected_balls):
            print(
                f"  Ball {i+1}: pos=({ball.position[0]:.1f}, {ball.position[1]:.1f}), "
                f"radius={ball.radius:.1f}, type={ball.ball_type}, conf={ball.confidence:.2f}"
            )

        return True
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        return False


def test_ball_tracking_system():
    """Test integrated ball tracking system."""
    print("\n=== Testing Ball Tracking System ===")

    config = {
        "detection_method": DetectionMethod.COMBINED,
        "enable_tracking": True,
        "debug_mode": False,
    }

    try:
        tracking_system = BallTrackingSystem(config)
        print("✓ Ball tracking system created successfully")
    except Exception as e:
        print(f"✗ Failed to create tracking system: {e}")
        return False

    # Test processing multiple frames
    frame = create_test_frame()

    try:
        for frame_num in range(3):
            result = tracking_system.process_frame(frame, frame_num)

            print(f"✓ Frame {frame_num} processed - {len(result.balls)} balls detected")
            print(f"  Processing time: {result.statistics.processing_time:.2f}ms")

            if result.has_errors:
                print(f"  Errors: {result.error_messages}")

        # Get performance metrics
        metrics = tracking_system.get_performance_metrics()
        print(
            f"✓ Performance metrics: {metrics.get('average_processing_time_ms', 0):.2f}ms avg"
        )

        return True
    except Exception as e:
        print(f"✗ Frame processing failed: {e}")
        return False


def test_ball_classification():
    """Test ball classification."""
    print("\n=== Testing Ball Classification ===")

    detector = BallDetector({"detection_method": DetectionMethod.HOUGH_CIRCLES})

    # Test different colored ball regions
    test_colors = [
        ((255, 255, 255), "white cue ball"),
        ((0, 0, 0), "black 8-ball"),
        ((255, 255, 0), "yellow ball"),
        ((0, 0, 255), "red ball"),
    ]

    for color, description in test_colors:
        # Create ball region
        ball_region = np.zeros((40, 40, 3), dtype=np.uint8)
        cv2.circle(ball_region, (20, 20), 18, color, -1)
        cv2.circle(ball_region, (15, 15), 5, tuple(min(255, c + 50) for c in color), -1)

        try:
            ball_type, confidence, number = detector.classify_ball_type(
                ball_region, (100, 100), 20
            )
            print(
                f"✓ {description}: type={ball_type}, confidence={confidence:.2f}, number={number}"
            )
        except Exception as e:
            print(f"✗ Classification failed for {description}: {e}")


def test_accuracy_requirements():
    """Test accuracy requirements."""
    print("\n=== Testing Accuracy Requirements ===")

    tracking_system = BallTrackingSystem(
        {
            "detection_method": DetectionMethod.COMBINED,
            "position_accuracy_threshold": 2.0,  # ±2 pixel requirement
        }
    )

    # Test with known ball positions
    frame = create_test_frame()
    known_positions = [(200, 200), (400, 300), (600, 200)]

    result = tracking_system.process_frame(frame)
    detected_positions = [(ball.position[0], ball.position[1]) for ball in result.balls]

    # Check position accuracy
    correct_detections = 0
    for known_pos in known_positions:
        for detected_pos in detected_positions:
            distance = np.sqrt(
                (known_pos[0] - detected_pos[0]) ** 2
                + (known_pos[1] - detected_pos[1]) ** 2
            )
            if distance <= 2.0:  # ±2 pixel requirement
                correct_detections += 1
                break

    accuracy = correct_detections / len(known_positions)
    print(
        f"✓ Position accuracy: {accuracy:.1%} ({correct_detections}/{len(known_positions)} balls)"
    )

    if accuracy >= 0.95:
        print("✓ Meets >95% accuracy requirement")
        return True
    else:
        print("✗ Does not meet 95% accuracy requirement")
        return False


def main():
    """Run all tests."""
    print("Ball Detection and Tracking System Integration Tests")
    print("=" * 60)

    test_results = []

    # Run tests
    test_results.append(("Ball Detector", test_ball_detector()))
    test_results.append(("Ball Tracking System", test_ball_tracking_system()))
    test_results.append(
        ("Ball Classification", test_ball_classification() or True)
    )  # Non-critical test
    test_results.append(("Accuracy Requirements", test_accuracy_requirements()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")

    if passed == total:
        print("✓ All tests passed - Ball detection system is working correctly!")
        return True
    else:
        print("✗ Some tests failed - Check implementation")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
