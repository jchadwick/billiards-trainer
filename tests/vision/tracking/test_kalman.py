"""Tests for Kalman filter implementation"""

import numpy as np
import pytest
from backend.vision.tracking.kalman import KalmanFilter, KalmanState


class TestKalmanFilter:
    """Test cases for KalmanFilter class"""

    def test_initialization(self):
        """Test Kalman filter initialization"""
        initial_pos = (100.0, 200.0)
        kf = KalmanFilter(initial_pos)

        assert kf.get_position() == initial_pos
        assert kf.get_velocity() == (0.0, 0.0)
        assert kf.get_acceleration() == (0.0, 0.0)
        assert kf.confidence == 1.0
        assert kf.age == 0

    def test_initialization_with_velocity(self):
        """Test initialization with initial velocity"""
        initial_pos = (100.0, 200.0)
        initial_vel = (10.0, -5.0)
        kf = KalmanFilter(initial_pos, initial_velocity=initial_vel)

        assert kf.get_position() == initial_pos
        assert kf.get_velocity() == initial_vel

    def test_predict_constant_velocity(self):
        """Test prediction with constant velocity"""
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

    def test_predict_with_acceleration(self):
        """Test prediction with acceleration"""
        initial_pos = (0.0, 0.0)
        initial_vel = (0.0, 0.0)
        initial_acc = (2.0, -1.0)
        kf = KalmanFilter(initial_pos, initial_velocity=initial_vel,
                         initial_acceleration=initial_acc)

        dt = 1.0
        predicted_pos = kf.predict(dt)

        # x = x0 + v0*t + 0.5*a*t^2
        expected_x = 0.0 + 0.0 * dt + 0.5 * 2.0 * dt**2
        expected_y = 0.0 + 0.0 * dt + 0.5 * (-1.0) * dt**2

        assert abs(predicted_pos[0] - expected_x) < 0.001
        assert abs(predicted_pos[1] - expected_y) < 0.001

    def test_update_measurement(self):
        """Test measurement update"""
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

    def test_predict_and_update_cycle(self):
        """Test complete predict-update cycle"""
        initial_pos = (0.0, 0.0)
        kf = KalmanFilter(initial_pos, process_noise=0.1, measurement_noise=1.0)

        # Simulate object moving in straight line
        true_positions = [(t, 2*t) for t in range(1, 6)]
        dt = 1.0

        for true_pos in true_positions:
            # Predict
            predicted_pos = kf.predict(dt)

            # Add some noise to simulate measurement
            noise = np.random.normal(0, 0.5, 2)
            measured_pos = (true_pos[0] + noise[0], true_pos[1] + noise[1])

            # Update
            kf.update(measured_pos)

        # Check that filter learned the velocity
        final_velocity = kf.get_velocity()
        assert abs(final_velocity[0] - 1.0) < 0.5  # dx/dt = 1
        assert abs(final_velocity[1] - 2.0) < 0.5  # dy/dt = 2

    def test_trajectory_prediction(self):
        """Test trajectory prediction"""
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

    def test_measurement_validation(self):
        """Test measurement validation"""
        initial_pos = (100.0, 100.0)
        kf = KalmanFilter(initial_pos, measurement_noise=1.0)

        # Valid measurement (close to prediction)
        assert kf.is_valid_measurement((102.0, 98.0))

        # Invalid measurement (far from prediction)
        assert not kf.is_valid_measurement((200.0, 200.0))

    def test_confidence_decay(self):
        """Test confidence decay over time"""
        initial_pos = (100.0, 100.0)
        kf = KalmanFilter(initial_pos)

        initial_confidence = kf.confidence

        # Predict without measurements
        for _ in range(10):
            kf.predict(0.1)

        # Confidence should decay
        assert kf.confidence < initial_confidence

    def test_quality_score(self):
        """Test quality score calculation"""
        initial_pos = (100.0, 100.0)
        kf = KalmanFilter(initial_pos)

        # Initially should have high quality
        initial_quality = kf.quality_score
        assert initial_quality > 0.5

        # Add some measurements
        for i in range(5):
            kf.predict(0.1)
            kf.update((100.0 + i, 100.0 + i))

        # Quality should remain high with good measurements
        assert kf.quality_score >= initial_quality * 0.8

    def test_reset_with_position(self):
        """Test resetting filter with new position"""
        initial_pos = (100.0, 100.0)
        kf = KalmanFilter(initial_pos, initial_velocity=(10.0, 5.0))

        # Let it evolve
        kf.predict(1.0)
        kf.update((110.0, 105.0))

        original_velocity = kf.get_velocity()

        # Reset position
        new_pos = (200.0, 300.0)
        kf.reset_with_position(new_pos)

        # Position should be reset, velocity preserved
        assert kf.get_position() == new_pos
        # Velocity should be similar (some uncertainty added)
        current_velocity = kf.get_velocity()
        assert abs(current_velocity[0] - original_velocity[0]) < 5.0
        assert abs(current_velocity[1] - original_velocity[1]) < 5.0

    def test_get_state(self):
        """Test getting complete filter state"""
        initial_pos = (100.0, 100.0)
        initial_vel = (5.0, -3.0)
        kf = KalmanFilter(initial_pos, initial_velocity=initial_vel)

        state = kf.get_state()

        assert isinstance(state, KalmanState)
        assert state.position == initial_pos
        assert state.velocity == initial_vel
        assert state.confidence == kf.confidence
        assert isinstance(state.covariance, np.ndarray)

    def test_covariance_matrices(self):
        """Test covariance matrix retrieval"""
        initial_pos = (100.0, 100.0)
        kf = KalmanFilter(initial_pos)

        pos_cov = kf.get_position_covariance()
        vel_cov = kf.get_velocity_covariance()

        assert pos_cov.shape == (2, 2)
        assert vel_cov.shape == (2, 2)
        assert np.all(np.diag(pos_cov) > 0)  # Positive diagonal
        assert np.all(np.diag(vel_cov) > 0)

    def test_speed_calculation(self):
        """Test speed calculation"""
        initial_pos = (0.0, 0.0)
        initial_vel = (3.0, 4.0)  # Speed should be 5.0
        kf = KalmanFilter(initial_pos, initial_velocity=initial_vel)

        speed = kf.get_speed()
        expected_speed = np.sqrt(3.0**2 + 4.0**2)
        assert abs(speed - expected_speed) < 0.001

    def test_numerical_stability(self):
        """Test numerical stability with edge cases"""
        # Test with very small noise
        kf1 = KalmanFilter((0.0, 0.0), process_noise=1e-10,
                          measurement_noise=1e-10)
        kf1.predict(0.1)
        kf1.update((0.1, 0.1))
        assert not np.any(np.isnan(kf1.state))

        # Test with very large noise
        kf2 = KalmanFilter((0.0, 0.0), process_noise=1e6,
                          measurement_noise=1e6)
        kf2.predict(0.1)
        kf2.update((100.0, 100.0))
        assert not np.any(np.isnan(kf2.state))

    @pytest.mark.parametrize("dt", [0.001, 0.1, 1.0, 10.0])
    def test_different_time_steps(self, dt):
        """Test filter with different time steps"""
        initial_pos = (0.0, 0.0)
        initial_vel = (10.0, -5.0)
        kf = KalmanFilter(initial_pos, initial_velocity=initial_vel)

        predicted_pos = kf.predict(dt)

        expected_x = 10.0 * dt
        expected_y = -5.0 * dt

        # Allow for some numerical error
        assert abs(predicted_pos[0] - expected_x) < 0.01
        assert abs(predicted_pos[1] - expected_y) < 0.01
