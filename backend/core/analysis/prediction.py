"""Outcome prediction algorithms."""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..models import BallState, GameState, ShotType, TableState, Vector2D
from ..physics.trajectory import TrajectoryCalculator
from ..utils.geometry import GeometryUtils


class OutcomeType(Enum):
    """Types of shot outcomes."""

    SUCCESS = "success"  # Target ball pocketed
    MISS = "miss"  # Target ball missed or not pocketed
    SCRATCH = "scratch"  # Cue ball pocketed
    FOUL = "foul"  # Rules violation
    SAFETY = "safety"  # Defensive outcome achieved
    COMBINATION = "combination"  # Multiple balls involved
    BANK = "bank"  # Cushion bounce shot


@dataclass
class PredictedOutcome:
    """Predicted outcome of a shot."""

    outcome_type: OutcomeType
    probability: float  # 0.0 to 1.0
    description: str
    final_ball_positions: dict[str, Vector2D]
    ball_trajectories: dict[str, list[Vector2D]]
    collision_sequence: list[dict[str, Any]]
    time_to_completion: float  # seconds
    confidence: float = 1.0  # Prediction confidence

    # Detailed outcome data
    target_ball_pocketed: bool = False
    cue_ball_pocketed: bool = False
    other_balls_pocketed: list[str] = field(default_factory=list)
    fouls_committed: list[str] = field(default_factory=list)

    # Physics data
    total_energy_dissipated: float = 0.0
    max_velocity_achieved: float = 0.0
    cushion_contacts: int = 0


@dataclass
class ShotPrediction:
    """Complete shot prediction with multiple possible outcomes."""

    primary_outcome: PredictedOutcome
    alternative_outcomes: list[PredictedOutcome]
    overall_confidence: float
    prediction_factors: dict[str, float]
    computational_time: float = 0.0

    # ML enhancement hooks
    ml_features: Optional[dict[str, Any]] = None
    ml_confidence: Optional[float] = None


class OutcomePredictor:
    """Game outcome prediction engine."""

    def __init__(self):
        """Initialize the outcome predictor with physics calculators and simulation parameters."""
        self.trajectory_calculator = TrajectoryCalculator()
        self.geometry_utils = GeometryUtils()

        # Prediction parameters
        self.max_simulation_time = 10.0  # seconds
        self.time_step = 0.01  # seconds
        self.velocity_threshold = 0.05  # m/s minimum velocity
        self.uncertainty_factor = 0.1  # Base uncertainty in predictions

        # Physics parameters for prediction
        self.air_resistance = 0.001
        self.rolling_friction = 0.2
        self.sliding_friction = 0.3
        self.spin_decay_rate = 0.1

    def predict_shot_outcome(
        self, game_state: GameState, shot_analysis
    ) -> ShotPrediction:
        """Predict the complete outcome of a shot."""
        # Extract shot parameters
        cue_ball = self._get_cue_ball(game_state)
        target_ball = self._get_ball_by_id(game_state, shot_analysis.target_ball_id)

        if not cue_ball or not target_ball:
            raise ValueError("Missing cue ball or target ball")

        # Create initial shot velocity vector
        shot_velocity = self._calculate_shot_velocity(
            shot_analysis.recommended_force, shot_analysis.recommended_angle
        )

        # Run physics simulation
        simulation_result = self._simulate_shot_physics(
            game_state, cue_ball, shot_velocity, shot_analysis
        )

        # Analyze outcomes
        primary_outcome = self._analyze_primary_outcome(
            game_state, simulation_result, shot_analysis
        )

        # Generate alternative outcomes based on uncertainties
        alternative_outcomes = self._generate_alternative_outcomes(
            game_state, shot_analysis, simulation_result
        )

        # Calculate overall confidence
        overall_confidence = self._calculate_prediction_confidence(
            game_state, shot_analysis, simulation_result
        )

        # Compile prediction factors
        prediction_factors = self._compile_prediction_factors(
            game_state, shot_analysis, simulation_result
        )

        computational_time = 0.0  # Would be time.time() - start_time

        return ShotPrediction(
            primary_outcome=primary_outcome,
            alternative_outcomes=alternative_outcomes,
            overall_confidence=overall_confidence,
            prediction_factors=prediction_factors,
            computational_time=computational_time,
        )

    def predict_outcomes(
        self, game_state: GameState, time_horizon: float = 5.0
    ) -> list[PredictedOutcome]:
        """Predict ball movements for time horizon."""
        outcomes = []

        # Get all moving balls
        moving_balls = [
            ball
            for ball in game_state.balls
            if not ball.is_pocketed and ball.is_moving()
        ]

        if not moving_balls:
            return []  # No motion to predict

        # Simulate each ball's trajectory
        for ball in moving_balls:
            trajectory = self.trajectory_calculator.calculate_path(
                ball, game_state.table, time_horizon
            )

            # Create outcome based on trajectory
            outcome = PredictedOutcome(
                outcome_type=(
                    OutcomeType.SUCCESS
                    if trajectory.will_be_pocketed
                    else OutcomeType.MISS
                ),
                probability=0.9 if trajectory.will_be_pocketed else 0.1,
                description=f"Ball {ball.id} prediction",
                final_ball_positions={ball.id: Vector2D(*trajectory.final_position)},
                ball_trajectories={
                    ball.id: [Vector2D(*pt) for pt in trajectory.points]
                },
                collision_sequence=[],
                time_to_completion=trajectory.time_to_rest,
                target_ball_pocketed=trajectory.will_be_pocketed,
            )

            outcomes.append(outcome)

        return outcomes

    def calculate_success_probability(self, shot_analysis) -> float:
        """Calculate probability of shot success."""
        # Base probability from shot analysis
        base_prob = shot_analysis.success_probability

        # Adjust based on shot complexity
        complexity_factor = 1.0
        if shot_analysis.shot_type == ShotType.BANK:
            complexity_factor = 0.8
        elif shot_analysis.shot_type == ShotType.COMBINATION:
            complexity_factor = 0.7
        elif shot_analysis.shot_type == ShotType.MASSE:
            complexity_factor = 0.5

        # Adjust based on identified problems
        problem_penalty = len(shot_analysis.potential_problems) * 0.1

        # Adjust based on risk factors
        risk_penalty = sum(shot_analysis.risk_factors.values()) * 0.1

        final_prob = base_prob * complexity_factor - problem_penalty - risk_penalty
        return max(0.05, min(0.95, final_prob))

    def _simulate_shot_physics(
        self,
        game_state: GameState,
        cue_ball: BallState,
        shot_velocity: Vector2D,
        shot_analysis,
    ) -> dict[str, Any]:
        """Run physics simulation of the shot."""
        # Create working copies of balls
        balls = [ball.copy() for ball in game_state.balls if not ball.is_pocketed]

        # Set cue ball velocity
        for ball in balls:
            if ball.is_cue_ball:
                ball.velocity = shot_velocity
                break

        # Simulation state
        sim_time = 0.0
        collision_sequence = []
        ball_trajectories = {ball.id: [ball.position] for ball in balls}
        energy_log = []

        # Main simulation loop
        while sim_time < self.max_simulation_time:
            # Check if all balls are at rest
            if all(not ball.is_moving(self.velocity_threshold) for ball in balls):
                break

            # Update ball positions
            self._update_ball_positions(balls, self.time_step, game_state.table)

            # Check for collisions
            collisions = self._detect_collisions(balls, game_state.table)

            # Process collisions
            for collision in collisions:
                self._process_collision(balls, collision)
                collision_sequence.append(
                    {
                        "time": sim_time,
                        "type": collision["type"],
                        "balls": collision.get("balls", []),
                        "position": collision.get("position", Vector2D(0, 0)).to_dict(),
                    }
                )

            # Record trajectories
            for ball in balls:
                ball_trajectories[ball.id].append(ball.position)

            # Record energy
            total_energy = sum(ball.kinetic_energy() for ball in balls)
            energy_log.append(total_energy)

            sim_time += self.time_step

        # Final ball positions
        final_positions = {ball.id: ball.position for ball in balls}

        return {
            "final_positions": final_positions,
            "ball_trajectories": ball_trajectories,
            "collision_sequence": collision_sequence,
            "simulation_time": sim_time,
            "energy_log": energy_log,
            "balls_pocketed": [ball.id for ball in balls if ball.is_pocketed],
        }

    def _analyze_primary_outcome(
        self, game_state: GameState, simulation_result: dict[str, Any], shot_analysis
    ) -> PredictedOutcome:
        """Analyze the primary predicted outcome."""
        target_ball_id = shot_analysis.target_ball_id

        # Check if target ball was pocketed
        target_pocketed = target_ball_id in simulation_result["balls_pocketed"]

        # Check if cue ball was pocketed (scratch)
        cue_ball_pocketed = any(
            ball.is_cue_ball
            for ball in game_state.balls
            if ball.id in simulation_result["balls_pocketed"]
        )

        # Determine outcome type
        if cue_ball_pocketed:
            outcome_type = OutcomeType.SCRATCH
            probability = 0.95  # High confidence in scratch detection
            description = "Cue ball scratch predicted"
        elif target_pocketed:
            outcome_type = OutcomeType.SUCCESS
            probability = self.calculate_success_probability(shot_analysis)
            description = f"Target ball {target_ball_id} pocketed"
        else:
            outcome_type = OutcomeType.MISS
            probability = 1.0 - self.calculate_success_probability(shot_analysis)
            description = f"Target ball {target_ball_id} missed"

        # Calculate additional metrics
        max_velocity = 0.0
        cushion_contacts = sum(
            1
            for collision in simulation_result["collision_sequence"]
            if collision["type"] == "cushion"
        )

        total_energy_dissipated = 0.0
        if simulation_result["energy_log"]:
            total_energy_dissipated = (
                simulation_result["energy_log"][0] - simulation_result["energy_log"][-1]
            )

        return PredictedOutcome(
            outcome_type=outcome_type,
            probability=probability,
            description=description,
            final_ball_positions=simulation_result["final_positions"],
            ball_trajectories=simulation_result["ball_trajectories"],
            collision_sequence=simulation_result["collision_sequence"],
            time_to_completion=simulation_result["simulation_time"],
            target_ball_pocketed=target_pocketed,
            cue_ball_pocketed=cue_ball_pocketed,
            other_balls_pocketed=[
                ball_id
                for ball_id in simulation_result["balls_pocketed"]
                if ball_id != target_ball_id
            ],
            total_energy_dissipated=total_energy_dissipated,
            max_velocity_achieved=max_velocity,
            cushion_contacts=cushion_contacts,
        )

    def _generate_alternative_outcomes(
        self, game_state: GameState, shot_analysis, simulation_result: dict[str, Any]
    ) -> list[PredictedOutcome]:
        """Generate alternative outcomes based on uncertainties."""
        alternatives = []

        # Uncertainty in shot power (±10%)
        for power_variation in [-0.1, 0.1]:
            alt_analysis = self._vary_shot_analysis(
                shot_analysis, power_factor=1.0 + power_variation
            )
            try:
                alt_result = self._simulate_shot_physics(
                    game_state,
                    self._get_cue_ball(game_state),
                    self._calculate_shot_velocity(
                        alt_analysis.recommended_force, alt_analysis.recommended_angle
                    ),
                    alt_analysis,
                )
                alt_outcome = self._analyze_primary_outcome(
                    game_state, alt_result, alt_analysis
                )
                alt_outcome.probability *= 0.3  # Lower probability for alternatives
                alt_outcome.confidence = 0.7
                alternatives.append(alt_outcome)
            except:
                continue

        # Uncertainty in shot angle (±2 degrees)
        for angle_variation in [-2.0, 2.0]:
            alt_analysis = self._vary_shot_analysis(
                shot_analysis, angle_offset=angle_variation
            )
            try:
                alt_result = self._simulate_shot_physics(
                    game_state,
                    self._get_cue_ball(game_state),
                    self._calculate_shot_velocity(
                        alt_analysis.recommended_force, alt_analysis.recommended_angle
                    ),
                    alt_analysis,
                )
                alt_outcome = self._analyze_primary_outcome(
                    game_state, alt_result, alt_analysis
                )
                alt_outcome.probability *= 0.2  # Lower probability for alternatives
                alt_outcome.confidence = 0.6
                alternatives.append(alt_outcome)
            except:
                continue

        # Sort by probability and return top alternatives
        alternatives.sort(key=lambda x: x.probability, reverse=True)
        return alternatives[:3]  # Return top 3 alternatives

    def _calculate_prediction_confidence(
        self, game_state: GameState, shot_analysis, simulation_result: dict[str, Any]
    ) -> float:
        """Calculate overall confidence in the prediction."""
        confidence_factors = []

        # Shot difficulty factor
        difficulty_confidence = 1.0 - shot_analysis.difficulty
        confidence_factors.append(difficulty_confidence)

        # Table condition factor
        table_confidence = 0.9 if game_state.table.surface_friction < 0.3 else 0.7
        confidence_factors.append(table_confidence)

        # Simulation stability factor (based on energy conservation)
        if simulation_result["energy_log"]:
            energy_stability = 1.0 - abs(
                simulation_result["energy_log"][-1]
                / max(simulation_result["energy_log"][0], 0.001)
            )
            confidence_factors.append(min(energy_stability, 1.0))

        # Collision complexity factor
        collision_complexity = max(
            0.5, 1.0 - len(simulation_result["collision_sequence"]) * 0.1
        )
        confidence_factors.append(collision_complexity)

        # Return average confidence
        return (
            sum(confidence_factors) / len(confidence_factors)
            if confidence_factors
            else 0.5
        )

    def _compile_prediction_factors(
        self, game_state: GameState, shot_analysis, simulation_result: dict[str, Any]
    ) -> dict[str, float]:
        """Compile factors that influence prediction accuracy."""
        return {
            "shot_difficulty": shot_analysis.difficulty,
            "table_friction": game_state.table.surface_friction,
            "collision_count": len(simulation_result["collision_sequence"]),
            "simulation_time": simulation_result["simulation_time"],
            "energy_dissipation": (
                simulation_result["energy_log"][0] - simulation_result["energy_log"][-1]
                if simulation_result["energy_log"]
                else 0.0
            ),
            "uncertainty_estimate": self.uncertainty_factor,
        }

    # Helper methods
    def _get_cue_ball(self, game_state: GameState) -> Optional[BallState]:
        """Get the cue ball from game state."""
        for ball in game_state.balls:
            if ball.is_cue_ball and not ball.is_pocketed:
                return ball
        return None

    def _get_ball_by_id(
        self, game_state: GameState, ball_id: str
    ) -> Optional[BallState]:
        """Get ball by ID from game state."""
        for ball in game_state.balls:
            if ball.id == ball_id and not ball.is_pocketed:
                return ball
        return None

    def _calculate_shot_velocity(self, force: float, angle_degrees: float) -> Vector2D:
        """Calculate velocity vector from force and angle."""
        angle_rad = math.radians(angle_degrees)
        # Convert force to velocity (simplified)
        velocity_magnitude = force * 0.1  # Simplified conversion factor
        return Vector2D(
            velocity_magnitude * math.cos(angle_rad),
            velocity_magnitude * math.sin(angle_rad),
        )

    def _vary_shot_analysis(
        self, original_analysis, power_factor: float = 1.0, angle_offset: float = 0.0
    ):
        """Create a varied version of shot analysis for uncertainty testing."""
        # Create a copy with variations
        import copy

        varied = copy.copy(original_analysis)
        varied.recommended_force *= power_factor
        varied.recommended_angle += angle_offset
        return varied

    def _update_ball_positions(
        self, balls: list[BallState], dt: float, table: TableState
    ):
        """Update ball positions based on physics."""
        for ball in balls:
            if not ball.is_moving(self.velocity_threshold):
                continue

            # Apply friction
            friction_force = self.rolling_friction * ball.mass * 9.81  # N
            friction_acceleration = friction_force / ball.mass  # m/s²

            # Reduce velocity due to friction
            velocity_magnitude = ball.velocity.magnitude()
            if velocity_magnitude > 0:
                friction_direction = ball.velocity.normalize() * -1
                velocity_change = friction_direction * friction_acceleration * dt
                new_velocity = ball.velocity + velocity_change

                # Don't reverse direction due to friction
                if new_velocity.dot(ball.velocity) < 0:
                    ball.velocity = Vector2D.zero()
                else:
                    ball.velocity = new_velocity

            # Update position
            ball.position = ball.position + (ball.velocity * dt)

            # Check table boundaries
            if not table.is_point_on_table(ball.position, ball.radius):
                # Simple boundary handling - this would be more sophisticated in reality
                if ball.position.x < ball.radius:
                    ball.position.x = ball.radius
                    ball.velocity.x *= -table.cushion_elasticity
                elif ball.position.x > table.width - ball.radius:
                    ball.position.x = table.width - ball.radius
                    ball.velocity.x *= -table.cushion_elasticity

                if ball.position.y < ball.radius:
                    ball.position.y = ball.radius
                    ball.velocity.y *= -table.cushion_elasticity
                elif ball.position.y > table.height - ball.radius:
                    ball.position.y = table.height - ball.radius
                    ball.velocity.y *= -table.cushion_elasticity

    def _detect_collisions(
        self, balls: list[BallState], table: TableState
    ) -> list[dict[str, Any]]:
        """Detect collisions between balls and with table elements."""
        collisions = []

        # Ball-ball collisions
        for i, ball1 in enumerate(balls):
            for ball2 in balls[i + 1 :]:
                if ball1.is_touching(ball2):
                    collisions.append(
                        {
                            "type": "ball",
                            "balls": [ball1.id, ball2.id],
                            "position": Vector2D(
                                (ball1.position.x + ball2.position.x) / 2,
                                (ball1.position.y + ball2.position.y) / 2,
                            ),
                        }
                    )

        # Ball-pocket collisions
        for ball in balls:
            is_in_pocket, pocket_index = table.is_point_in_pocket(ball.position)
            if is_in_pocket:
                ball.is_pocketed = True
                ball.velocity = Vector2D.zero()
                collisions.append(
                    {
                        "type": "pocket",
                        "balls": [ball.id],
                        "position": ball.position,
                        "pocket_index": pocket_index,
                    }
                )

        return collisions

    def _process_collision(self, balls: list[BallState], collision: dict[str, Any]):
        """Process a collision and update ball velocities."""
        if collision["type"] == "ball" and len(collision["balls"]) == 2:
            # Ball-ball collision
            ball1_id, ball2_id = collision["balls"]
            ball1 = next(b for b in balls if b.id == ball1_id)
            ball2 = next(b for b in balls if b.id == ball2_id)

            # Simple elastic collision (simplified)
            # In reality, this would involve more complex physics
            v1, v2 = ball1.velocity, ball2.velocity
            m1, m2 = ball1.mass, ball2.mass

            # Conservation of momentum and energy
            new_v1 = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
            new_v2 = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)

            ball1.velocity = new_v1
            ball2.velocity = new_v2

        elif collision["type"] == "pocket":
            # Ball pocketed - already handled in detection
            pass


# Alias for backward compatibility with tests
ShotPredictor = OutcomePredictor
