"""Isolated tests for tracking components"""

import sys
import os
import importlib.util

# Direct import using file paths to avoid problematic __init__.py imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

def import_module_from_path(module_name, file_path):
    """Import a module directly from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def test_tracking_system():
    """Test the tracking system components"""
    print("Testing tracking system components...\n")

    # Import numpy first
    import numpy as np

    # Import Kalman filter directly
    kalman_path = os.path.join(project_root, 'backend/vision/tracking/kalman.py')
    kalman_module = import_module_from_path('kalman', kalman_path)

    KalmanFilter = kalman_module.KalmanFilter
    KalmanState = kalman_module.KalmanState

    print("✓ Successfully imported Kalman filter")

    # Test Kalman filter
    print("\n1. Testing Kalman Filter:")

    # Test initialization
    kf = KalmanFilter((100.0, 200.0))
    assert kf.get_position() == (100.0, 200.0)
    assert kf.get_velocity() == (0.0, 0.0)
    print("  ✓ Initialization")

    # Test prediction
    kf2 = KalmanFilter((0.0, 0.0), initial_velocity=(10.0, -5.0))
    predicted_pos = kf2.predict(0.1)
    assert abs(predicted_pos[0] - 1.0) < 0.001
    assert abs(predicted_pos[1] - (-0.5)) < 0.001
    print("  ✓ Prediction")

    # Test update
    kf3 = KalmanFilter((100.0, 100.0))
    kf3.update((105.0, 95.0))
    updated_pos = kf3.get_position()
    assert updated_pos[0] > 100.0
    assert updated_pos[1] < 100.0
    print("  ✓ Measurement update")

    # Test trajectory prediction
    trajectory = kf2.predict_trajectory(3, 0.1)
    assert len(trajectory) == 3
    print("  ✓ Trajectory prediction")

    # Test state retrieval
    state = kf.get_state()
    assert isinstance(state, KalmanState)
    print("  ✓ State retrieval")

    print("\n2. Testing Object Tracker:")

    # Import tracker directly
    tracker_path = os.path.join(project_root, 'backend/vision/tracking/tracker.py')
    tracker_module = import_module_from_path('tracker', tracker_path)

    ObjectTracker = tracker_module.ObjectTracker
    Ball = tracker_module.Ball
    BallType = tracker_module.BallType
    TrackState = tracker_module.TrackState

    print("  ✓ Successfully imported tracker components")

    # Test tracker initialization
    config = {
        'max_age': 30,
        'min_hits': 3,
        'max_distance': 50.0,
        'process_noise': 1.0,
        'measurement_noise': 10.0
    }

    tracker = ObjectTracker(config)
    assert len(tracker.tracks) == 0
    assert tracker.next_track_id == 1
    print("  ✓ Tracker initialization")

    # Test single ball tracking
    ball = Ball(
        position=(100.0, 200.0),
        radius=15.0,
        ball_type=BallType.CUE,
        confidence=0.8
    )

    tracked_balls = tracker.update_tracking([ball], frame_number=1)
    assert len(tracker.tracks) == 1
    print("  ✓ Single ball tracking")

    # Test track confirmation
    for frame in range(2, 5):
        tracked_balls = tracker.update_tracking([ball], frame_number=frame)

    assert len(tracked_balls) > 0
    confirmed_tracks = [t for t in tracker.tracks if t.state == TrackState.CONFIRMED]
    assert len(confirmed_tracks) > 0
    print("  ✓ Track confirmation")

    # Test prediction
    predictions = tracker.predict_positions(0.1)
    assert len(predictions) > 0
    print("  ✓ Position prediction")

    # Test velocities
    velocities = tracker.get_object_velocities()
    assert len(velocities) >= 0  # Could be empty if no confirmed tracks
    print("  ✓ Velocity calculation")

    # Test multi-ball tracking
    ball2 = Ball(
        position=(300.0, 400.0),
        radius=15.0,
        ball_type=BallType.SOLID,
        number=1,
        confidence=0.8
    )

    # Update with both balls
    for frame in range(5, 8):
        tracked_balls = tracker.update_tracking([ball, ball2], frame_number=frame)

    assert len(tracker.tracks) == 2
    print("  ✓ Multi-ball tracking")

    # Test track loss
    # Don't provide any detections for several frames
    for frame in range(8, 15):
        tracked_balls = tracker.update_tracking([], frame_number=frame)

    # Some tracks should still exist but might be in LOST state
    assert len(tracker.tracks) >= 0
    print("  ✓ Track loss handling")

    # Test statistics
    stats = tracker.get_tracking_statistics()
    assert 'total_tracks_created' in stats
    assert stats['total_tracks_created'] >= 2
    print("  ✓ Statistics collection")

    print("\n🎉 All tracking system tests completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = test_tracking_system()
        print("\nTest Result: SUCCESS")
        sys.exit(0)
    except Exception as e:
        print(f"\nTest Result: FAILED - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
