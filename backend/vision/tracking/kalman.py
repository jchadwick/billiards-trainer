"""Kalman filter for position prediction."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class KalmanState:
    """Kalman filter state representation."""

    position: tuple[float, float]
    velocity: tuple[float, float]
    acceleration: tuple[float, float]
    covariance: np.ndarray
    confidence: float


class KalmanFilter:
    """Kalman filter for object position, velocity, and acceleration tracking.

    State vector: [x, y, vx, vy, ax, ay]
    - Position: (x, y)
    - Velocity: (vx, vy)
    - Acceleration: (ax, ay)
    """

    def __init__(
        self,
        initial_position: tuple[float, float],
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
        initial_velocity: Optional[tuple[float, float]] = None,
        initial_acceleration: Optional[tuple[float, float]] = None,
    ):
        """Initialize Kalman filter with initial state.

        Args:
            initial_position: Initial (x, y) position
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            initial_velocity: Initial velocity estimate (default: (0, 0))
            initial_acceleration: Initial acceleration estimate (default: (0, 0))
        """
        self.dim_state = 6  # [x, y, vx, vy, ax, ay]
        self.dim_obs = 2  # [x, y] observations

        # Initialize state vector
        self.state = np.zeros(self.dim_state)
        self.state[0:2] = initial_position
        if initial_velocity:
            self.state[2:4] = initial_velocity
        if initial_acceleration:
            self.state[4:6] = initial_acceleration

        # Initialize covariance matrix
        self.P = np.eye(self.dim_state) * 1000  # High initial uncertainty
        self.P[0:2, 0:2] *= 0.1  # Lower uncertainty for initial position

        # Process noise covariance (Q)
        self.Q = np.eye(self.dim_state) * process_noise
        self.Q[4:6, 4:6] *= 10  # Higher noise for acceleration

        # Measurement noise covariance (R)
        self.R = np.eye(self.dim_obs) * measurement_noise

        # Observation matrix (H) - we observe position only
        self.H = np.zeros((self.dim_obs, self.dim_state))
        self.H[0, 0] = 1  # x position
        self.H[1, 1] = 1  # y position

        # State transition matrix (F) - will be updated with dt
        self.F = np.eye(self.dim_state)

        # Track quality metrics
        self.innovation_covariance = np.eye(self.dim_obs)
        self.mahalanobis_distance = 0.0
        self.confidence = 1.0
        self.age = 0
        self.hit_streak = 0
        self.time_since_update = 0

    def predict(self, dt: float) -> tuple[float, float]:
        """Predict next position using constant acceleration model.

        Args:
            dt: Time delta since last update

        Returns:
            Predicted (x, y) position
        """
        # Update state transition matrix with time step
        self.F[0, 2] = dt  # x = x + vx*dt
        self.F[1, 3] = dt  # y = y + vy*dt
        self.F[0, 4] = 0.5 * dt**2  # x = x + 0.5*ax*dt^2
        self.F[1, 5] = 0.5 * dt**2  # y = y + 0.5*ay*dt^2
        self.F[2, 4] = dt  # vx = vx + ax*dt
        self.F[3, 5] = dt  # vy = vy + ay*dt

        # Predict state
        self.state = self.F @ self.state

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update tracking metrics
        self.age += 1
        self.time_since_update += 1

        # Decay confidence over time without measurements
        if self.time_since_update > 0:
            self.confidence *= 0.95

        return (self.state[0], self.state[1])

    def update(self, measured_position: tuple[float, float]) -> None:
        """Update filter with measured position using Kalman update equations.

        Args:
            measured_position: Observed (x, y) position
        """
        # Convert measurement to numpy array
        z = np.array(measured_position)

        # Innovation (prediction error)
        y = z - self.H @ self.state

        # Innovation covariance
        self.innovation_covariance = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(self.innovation_covariance)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I_KH = np.eye(self.dim_state) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Calculate quality metrics
        self.mahalanobis_distance = np.sqrt(
            y.T @ np.linalg.inv(self.innovation_covariance) @ y
        )

        # Update tracking metrics
        self.hit_streak += 1
        self.time_since_update = 0

        # Update confidence based on innovation
        innovation_factor = 1.0 / (1.0 + self.mahalanobis_distance / 10.0)
        self.confidence = min(1.0, self.confidence * 0.9 + innovation_factor * 0.1)

    def get_position(self) -> tuple[float, float]:
        """Get current estimated position."""
        return (self.state[0], self.state[1])

    def get_velocity(self) -> tuple[float, float]:
        """Get current estimated velocity."""
        return (self.state[2], self.state[3])

    def get_acceleration(self) -> tuple[float, float]:
        """Get current estimated acceleration."""
        return (self.state[4], self.state[5])

    def get_speed(self) -> float:
        """Get current speed (magnitude of velocity)."""
        vx, vy = self.get_velocity()
        return np.sqrt(vx**2 + vy**2)

    def get_state(self) -> KalmanState:
        """Get complete filter state."""
        return KalmanState(
            position=self.get_position(),
            velocity=self.get_velocity(),
            acceleration=self.get_acceleration(),
            covariance=self.P.copy(),
            confidence=self.confidence,
        )

    def get_position_covariance(self) -> np.ndarray:
        """Get position covariance matrix."""
        return self.P[0:2, 0:2]

    def get_velocity_covariance(self) -> np.ndarray:
        """Get velocity covariance matrix."""
        return self.P[2:4, 2:4]

    def is_valid_measurement(
        self, position: tuple[float, float], max_mahalanobis: float = 9.0
    ) -> bool:
        """Check if a measurement is valid based on Mahalanobis distance.

        Args:
            position: Measured position
            max_mahalanobis: Maximum allowed Mahalanobis distance

        Returns:
            True if measurement is valid
        """
        z = np.array(position)
        predicted_pos = self.H @ self.state
        innovation = z - predicted_pos

        # Use current innovation covariance or predict it
        S = self.H @ self.P @ self.H.T + self.R

        try:
            mahal_dist = np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation)
            return mahal_dist < max_mahalanobis
        except np.linalg.LinAlgError:
            # If covariance is singular, be conservative
            return False

    def predict_trajectory(self, time_steps: int, dt: float) -> list:
        """Predict future trajectory for multiple time steps.

        Args:
            time_steps: Number of future steps to predict
            dt: Time delta for each step

        Returns:
            List of predicted (x, y) positions
        """
        # Create a temporary copy to avoid modifying current state
        temp_state = self.state.copy()
        temp_F = self.F.copy()

        # Update transition matrix
        temp_F[0, 2] = dt
        temp_F[1, 3] = dt
        temp_F[0, 4] = 0.5 * dt**2
        temp_F[1, 5] = 0.5 * dt**2
        temp_F[2, 4] = dt
        temp_F[3, 5] = dt

        trajectory = []
        for _ in range(time_steps):
            temp_state = temp_F @ temp_state
            trajectory.append((temp_state[0], temp_state[1]))

        return trajectory

    def reset_with_position(self, position: tuple[float, float]) -> None:
        """Reset filter with new position while preserving some state.

        Args:
            position: New position to reset with
        """
        # Keep velocity estimates but reset position
        self.state[0:2] = position

        # Reset position covariance but keep velocity covariance
        self.P[0:2, 0:2] = np.eye(2) * 100

        # Reset tracking metrics
        self.time_since_update = 0
        self.hit_streak = 0
        self.confidence = min(0.8, self.confidence)

    @property
    def quality_score(self) -> float:
        """Calculate overall track quality score.

        Returns:
            Quality score between 0 and 1
        """
        # Base score on confidence
        score = self.confidence

        # Adjust based on age and hit streak
        if self.age > 0:
            hit_ratio = self.hit_streak / self.age
            score *= hit_ratio

        # Penalize tracks without recent updates
        if self.time_since_update > 5:
            score *= 0.5

        # Penalize high uncertainty
        pos_uncertainty = np.trace(self.get_position_covariance())
        uncertainty_factor = 1.0 / (1.0 + pos_uncertainty / 1000.0)
        score *= uncertainty_factor

        return max(0.0, min(1.0, score))
