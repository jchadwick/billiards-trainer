"""Simple trajectory prediction for billiards shots.

All spatial calculations use the 4K coordinate system (3840×2160).
Physics simulations and trajectory predictions operate in 4K pixel space.

Simplified predictions include:
- Fixed medium power (no variable power)
- No spin/English effects
- 2-level collision detection max (cue ball hits → those balls hit)
- Straight cue angle through cue ball center
"""

from typing import Optional

from config import Config, config

from ..coordinates import Vector2D
from ..models import BallState, CueState, GameState
from ..physics.trajectory import TrajectoryCalculator


class TrajectoryPredictor:
    """Simple trajectory prediction for cue shots."""

    # Fixed medium shot speed (m/s) - no variable power
    MEDIUM_SHOT_SPEED = 2.0

    def __init__(self, cfg: Optional[Config] = None):
        """Initialize the trajectory predictor."""
        self.trajectory_calculator = TrajectoryCalculator()
        self.config = cfg if cfg is not None else config

        # Get configuration
        self.max_simulation_time = self.config.get(
            "core.prediction.max_prediction_time", 10.0
        )
        self.max_collision_depth = 2  # Hardcoded: max 2 collision levels

    def predict_cue_shot(
        self,
        cue_state: CueState,
        game_state: GameState,
    ) -> dict[str, list[Vector2D]]:
        """Predict trajectory for a cue shot.

        Args:
            cue_state: Cue stick state (position and angle)
            game_state: Current game state (balls, table)

        Returns:
            Dictionary mapping ball IDs to their trajectory points:
            {
                "cue": [Vector2D(...), Vector2D(...), ...],
                "ball_1": [Vector2D(...), ...],
                "ball_2": [...],
            }
        """
        # Get cue ball
        cue_ball = self._get_cue_ball(game_state)
        if not cue_ball:
            return {}

        # Calculate initial velocity from cue angle
        # Fixed medium power, straight through cue ball center
        initial_velocity = self._calculate_shot_velocity(cue_state.angle)

        # Create a working copy of the cue ball with velocity
        working_cue_ball = cue_ball.copy()
        working_cue_ball.velocity = initial_velocity

        # Get other balls (excluding cue ball)
        other_balls = [
            ball.copy()
            for ball in game_state.balls
            if not ball.is_cue_ball and not ball.is_pocketed
        ]

        # Calculate trajectories with max 2 collision levels
        # Note: We need to create a minimal CueState from the cue_state parameter
        multiball_result = self.trajectory_calculator.predict_multiball_cue_shot(
            cue_state=cue_state,
            ball_state=working_cue_ball,
            table_state=game_state.table,
            other_balls=other_balls,
            max_collision_depth=self.max_collision_depth,
        )

        # Convert MultiballTrajectoryResult to simple dict of trajectories
        # Extract just the trajectory points for each ball
        trajectories = {}
        for ball_id, trajectory in multiball_result.trajectories.items():
            trajectories[ball_id] = [point.position for point in trajectory.points]

        return trajectories

    def _get_cue_ball(self, game_state: GameState) -> Optional[BallState]:
        """Find and return the cue ball from game state."""
        for ball in game_state.balls:
            if ball.is_cue_ball and not ball.is_pocketed:
                return ball
        return None

    def _calculate_shot_velocity(self, cue_angle_degrees: float) -> Vector2D:
        """Calculate velocity vector from cue angle.

        Args:
            cue_angle_degrees: Cue stick angle in degrees

        Returns:
            Velocity vector in 4K coordinates (pixels/second)
        """
        import math

        # Convert angle to radians
        angle_rad = math.radians(cue_angle_degrees)

        # Calculate velocity components
        # Fixed medium speed, direction from cue angle
        vx = self.MEDIUM_SHOT_SPEED * math.cos(angle_rad)
        vy = self.MEDIUM_SHOT_SPEED * math.sin(angle_rad)

        # Return as 4K coordinate vector
        return Vector2D.from_meters(vx, vy)


# Backward compatibility alias
ShotPredictor = TrajectoryPredictor
