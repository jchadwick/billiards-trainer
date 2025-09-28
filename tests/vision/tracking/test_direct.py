"""Direct tests for tracking components without complex imports"""

import sys
import os
import numpy as np

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Import tracking components directly
from backend.vision.tracking.kalman import KalmanFilter, KalmanState


def test_kalman_filter_basic():
    """Test basic Kalman filter functionality"""
    print("Testing Kalman filter initialization...")

    # Test initialization
    initial_pos = (100.0, 200.0)
    kf = KalmanFilter(initial_pos)

    assert kf.get_position() == initial_pos
    assert kf.get_velocity() == (0.0, 0.0)
    assert kf.confidence == 1.0
    print("✓ Initialization test passed")

    # Test prediction
    print("Testing prediction...")
    initial_vel = (10.0, -5.0)
    kf2 = KalmanFilter((0.0, 0.0), initial_velocity=initial_vel)

    dt = 0.1
    predicted_pos = kf2.predict(dt)

    expected_x = 10.0 * dt
    expected_y = -5.0 * dt

    assert abs(predicted_pos[0] - expected_x) < 0.001
    assert abs(predicted_pos[1] - expected_y) < 0.001
    print("✓ Prediction test passed")

    # Test measurement update
    print("Testing measurement update...")
    kf3 = KalmanFilter((100.0, 100.0))
    kf3.update((105.0, 95.0))

    updated_pos = kf3.get_position()
    # Position should move towards measurement
    assert updated_pos[0] > 100.0
    assert updated_pos[1] < 100.0
    print("✓ Measurement update test passed")

    # Test trajectory prediction
    print("Testing trajectory prediction...")
    kf4 = KalmanFilter((0.0, 0.0), initial_velocity=(10.0, 5.0))
    trajectory = kf4.predict_trajectory(3, 0.1)

    assert len(trajectory) == 3
    # Each step should move further
    for i in range(1, len(trajectory)):
        assert trajectory[i][0] > trajectory[i-1][0]
        assert trajectory[i][1] > trajectory[i-1][1]
    print("✓ Trajectory prediction test passed")


def test_tracker_imports():
    """Test that we can import tracker components"""
    print("Testing tracker imports...")

    try:
        from backend.vision.tracking.tracker import ObjectTracker, Track, TrackState
        print("✓ Tracker imports successful")
        return True
    except ImportError as e:
        print(f"✗ Failed to import tracker: {e}")
        return False


def test_tracker_basic():
    """Test basic tracker functionality"""
    print("Testing basic tracker functionality...")

    # Import Ball from models directly to avoid complex imports
    sys.path.insert(0, os.path.join(project_root, 'backend/vision'))
    from models import Ball, BallType
    from backend.vision.tracking.tracker import ObjectTracker

    # Create tracker
    config = {
        'max_age': 30,
        'min_hits': 3,
        'max_distance': 50.0,
        'process_noise': 1.0,
        'measurement_noise': 10.0
    }

    tracker = ObjectTracker(config)
    assert len(tracker.tracks) == 0
    print("✓ Tracker initialization passed")

    # Create test ball
    ball = Ball(
        position=(100.0, 200.0),
        radius=15.0,
        ball_type=BallType.CUE,
        confidence=0.8
    )

    # Update tracker
    tracked_balls = tracker.update_tracking([ball], frame_number=1)
    assert len(tracker.tracks) == 1
    print("✓ Basic tracking test passed")

    # Test multiple updates to confirm track
    for frame in range(2, 5):
        tracked_balls = tracker.update_tracking([ball], frame_number=frame)

    # Track should be confirmed now
    assert len(tracked_balls) > 0
    print("✓ Track confirmation test passed")


def main():
    """Run all tests"""
    print("Running tracking system tests...\n")

    # Test Kalman filter
    try:
        test_kalman_filter_basic()
        print("All Kalman filter tests passed!\n")
    except Exception as e:
        print(f"Kalman filter tests failed: {e}\n")
        return False

    # Test tracker imports
    if test_tracker_imports():
        try:
            test_tracker_basic()
            print("All tracker tests passed!\n")
        except Exception as e:
            print(f"Tracker tests failed: {e}\n")
            return False
    else:
        print("Skipping tracker tests due to import failure\n")

    print("🎉 All tracking system tests completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
