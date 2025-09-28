"""Simple tests for Kalman filter implementation"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import pytest


def test_kalman_imports():
    """Test that we can import Kalman filter components"""
    try:
        from backend.vision.tracking.kalman import KalmanFilter, KalmanState
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import Kalman components: {e}")


def test_kalman_initialization():
    """Test Kalman filter initialization"""
    from backend.vision.tracking.kalman import KalmanFilter

    initial_pos = (100.0, 200.0)
    kf = KalmanFilter(initial_pos)

    assert kf.get_position() == initial_pos
    assert kf.get_velocity() == (0.0, 0.0)
    assert kf.get_acceleration() == (0.0, 0.0)
    assert kf.confidence == 1.0
    assert kf.age == 0


def test_kalman_prediction():
    """Test Kalman filter prediction"""
    from backend.vision.tracking.kalman import KalmanFilter

    initial_pos = (100.0, 200.0)
    initial_vel = (10.0, -5.0)
    kf = KalmanFilter(initial_pos, initial_velocity=initial_vel)

    dt = 0.1
    predicted_pos = kf.predict(dt)

    expected_x = 100.0 + 10.0 * dt
    expected_y = 200.0 + (-5.0) * dt

    assert abs(predicted_pos[0] - expected_x) < 0.001
    assert abs(predicted_pos[1] - expected_y) < 0.001
    assert kf.age == 1


def test_kalman_update():
    """Test Kalman filter measurement update"""
    from backend.vision.tracking.kalman import KalmanFilter

    initial_pos = (100.0, 100.0)
    kf = KalmanFilter(initial_pos, measurement_noise=1.0)

    # Make a measurement
    measured_pos = (105.0, 95.0)
    kf.update(measured_pos)

    # Position should be adjusted towards measurement
    updated_pos = kf.get_position()
    assert 100.0 < updated_pos[0] < 105.0
    assert 95.0 < updated_pos[1] < 100.0
    assert kf.time_since_update == 0


def test_kalman_trajectory_prediction():
    """Test trajectory prediction"""
    from backend.vision.tracking.kalman import KalmanFilter

    initial_pos = (0.0, 0.0)
    initial_vel = (10.0, 5.0)
    kf = KalmanFilter(initial_pos, initial_velocity=initial_vel)

    # Predict 5 steps into future
    trajectory = kf.predict_trajectory(5, 0.1)

    assert len(trajectory) == 5
    # Check first prediction
    expected_x = 0.0 + 10.0 * 0.1
    expected_y = 0.0 + 5.0 * 0.1
    assert abs(trajectory[0][0] - expected_x) < 0.001
    assert abs(trajectory[0][1] - expected_y) < 0.001


if __name__ == "__main__":
    # Run basic tests
    test_kalman_imports()
    test_kalman_initialization()
    test_kalman_prediction()
    test_kalman_update()
    test_kalman_trajectory_prediction()
    print("All Kalman filter tests passed!")
