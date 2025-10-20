"""Shot analysis algorithms."""

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from backend.config import Config, config

from ..models import BallState, GameState, ShotType, TableState, Vector2D
from ..physics.trajectory import TrajectoryCalculator
from ..utils.geometry import GeometryUtils


def _get_config() -> Config:
    """Get the global configuration instance."""
    return config


class IllegalShotReason(Enum):
    """Reasons why a shot might be illegal."""

    WRONG_BALL_FIRST = "must_hit_lowest_numbered_ball_first"
    NO_CONTACT = "cue_ball_must_contact_object_ball"
    SCRATCH = "cue_ball_pocketed"
    JUMP_SHOT_ILLEGAL = "jump_shots_not_allowed"
    SAFETY_REQUIREMENTS = "safety_shot_requirements_not_met"
    EIGHT_BALL_EARLY = "eight_ball_hit_before_group_cleared"
    WRONG_POCKET = "eight_ball_called_wrong_pocket"


@dataclass
class ShotAnalysis:
    """Shot analysis and recommendations."""

    shot_type: ShotType
    difficulty: float  # 0.0 (easy) to 1.0 (hard)
    success_probability: float  # 0.0 to 1.0
    recommended_force: float  # Newtons
    recommended_angle: float  # degrees
    recommended_aim_point: Vector2D
    potential_problems: list[str]
    alternative_shots: list["ShotAnalysis"] = field(default_factory=list)
    is_legal: bool = True
    illegal_reasons: list[IllegalShotReason] = field(default_factory=list)
    target_ball_id: Optional[str] = None
    expected_outcome: Optional[str] = None
    risk_factors: dict[str, float] = field(default_factory=dict)

    # ML hooks for future enhancement
    ml_confidence: Optional[float] = None
    ml_features: Optional[dict[str, Any]] = None


class ShotAnalyzer:
    """Shot analysis engine."""

    def __init__(self, cfg: Optional[Config] = None):
        self.geometry_utils = GeometryUtils()
        self.trajectory_calculator = TrajectoryCalculator()
        self.config = config or _get_config()

        # Load shot difficulty factors weights from config
        self.difficulty_weights = {
            "distance": self.config.get(
                "core.shot_analysis.difficulty_weights.distance", 0.25
            ),
            "angle": self.config.get(
                "core.shot_analysis.difficulty_weights.angle", 0.20
            ),
            "obstacles": self.config.get(
                "core.shot_analysis.difficulty_weights.obstacles", 0.15
            ),
            "precision_required": self.config.get(
                "core.shot_analysis.difficulty_weights.precision_required", 0.20
            ),
            "cushion_bounces": self.config.get(
                "core.shot_analysis.difficulty_weights.cushion_bounces", 0.10
            ),
            "spin_required": self.config.get(
                "core.shot_analysis.difficulty_weights.spin_required", 0.10
            ),
        }

    def analyze_shot(
        self,
        game_state: GameState,
        target_ball: Optional[str] = None,
        aim_point: Optional[Vector2D] = None,
        force: Optional[float] = None,
    ) -> ShotAnalysis:
        """Analyze current shot setup."""
        cue_ball = self._get_cue_ball(game_state)
        if not cue_ball:
            raise ValueError("No cue ball found")

        # Determine target ball if not specified
        if not target_ball:
            target_ball = self._determine_best_target(game_state)

        target_ball_obj = self._get_ball_by_id(game_state, target_ball)
        if not target_ball_obj:
            raise ValueError(f"Target ball {target_ball} not found")

        # Determine shot type
        shot_type = self._identify_shot_type(
            game_state, cue_ball, target_ball_obj, aim_point
        )

        # Calculate basic shot parameters
        if not aim_point:
            aim_point = self._calculate_optimal_aim_point(
                cue_ball, target_ball_obj, game_state.table
            )

        if not force:
            force = self._calculate_recommended_force(
                cue_ball, target_ball_obj, shot_type
            )

        angle = self._calculate_shot_angle(cue_ball.position, aim_point)

        # Calculate difficulty
        difficulty = self.calculate_difficulty(
            game_state, cue_ball, target_ball_obj, shot_type
        )

        # Calculate success probability
        success_probability = self._calculate_success_probability(
            difficulty, shot_type, game_state
        )

        # Check legality
        is_legal, illegal_reasons = self._check_shot_legality(
            game_state, target_ball_obj
        )

        # Identify potential problems
        problems = self._identify_potential_problems(
            game_state, cue_ball, target_ball_obj, shot_type
        )

        # Calculate risk factors
        risk_factors = self._calculate_risk_factors(
            game_state, cue_ball, target_ball_obj, shot_type
        )

        # Predict expected outcome
        expected_outcome = self._predict_shot_outcome(
            game_state, cue_ball, target_ball_obj, aim_point, force, shot_type
        )

        return ShotAnalysis(
            shot_type=shot_type,
            difficulty=difficulty,
            success_probability=success_probability,
            recommended_force=force,
            recommended_angle=angle,
            recommended_aim_point=aim_point,
            potential_problems=problems,
            is_legal=is_legal,
            illegal_reasons=illegal_reasons,
            target_ball_id=target_ball,
            expected_outcome=expected_outcome,
            risk_factors=risk_factors,
        )

    def _identify_shot_type(
        self,
        game_state: GameState,
        cue_ball: BallState,
        target_ball: BallState,
        aim_point: Optional[Vector2D],
    ) -> ShotType:
        """Identify the type of shot being attempted."""
        # Check if it's a break shot
        if game_state.is_break:
            return ShotType.BREAK

        # Calculate direct path from cue ball to target
        self._is_direct_path_blocked(game_state, cue_ball, target_ball)

        # Check for bank shot - if aiming at cushion first
        if aim_point and self._is_cushion_shot(
            game_state, cue_ball.position, aim_point
        ):
            return ShotType.BANK

        # Check for combination shot - multiple balls involved
        if self._is_combination_shot(game_state, cue_ball, target_ball):
            return ShotType.COMBINATION

        # Check for safety shot - defensive play
        if self._is_safety_shot(game_state, target_ball):
            return ShotType.SAFETY

        # Check for masse shot - extreme spin/curve required
        if self._requires_masse(game_state, cue_ball, target_ball):
            return ShotType.MASSE

        # Default to direct shot
        return ShotType.DIRECT

    def calculate_difficulty(
        self,
        game_state: GameState,
        cue_ball: BallState,
        target_ball: BallState,
        shot_type: ShotType,
    ) -> float:
        """Calculate shot difficulty score (0.0 = easy, 1.0 = extremely hard)."""
        factors = {}

        # Distance factor (0.0 = very close, 1.0 = across table)
        distance = self.geometry_utils.distance_between_points(
            (cue_ball.position.x, cue_ball.position.y),
            (target_ball.position.x, target_ball.position.y),
        )
        max_distance = math.sqrt(game_state.table.width**2 + game_state.table.height**2)
        factors["distance"] = min(distance / max_distance, 1.0)

        # Angle factor - how thin the cut is
        factors["angle"] = self._calculate_angle_difficulty(
            cue_ball, target_ball, game_state.table
        )

        # Obstacles factor - interference from other balls
        factors["obstacles"] = self._calculate_obstacle_difficulty(
            game_state, cue_ball, target_ball
        )

        # Precision required factor - pocket size vs ball path
        factors["precision_required"] = self._calculate_precision_difficulty(
            target_ball, game_state.table
        )

        # Cushion bounces factor
        factors["cushion_bounces"] = self._calculate_cushion_difficulty(shot_type)

        # Spin required factor
        factors["spin_required"] = self._calculate_spin_difficulty(
            shot_type, cue_ball, target_ball
        )

        # Calculate weighted difficulty
        total_difficulty = sum(
            factors[factor] * self.difficulty_weights[factor] for factor in factors
        )

        # Apply shot type modifiers from config
        type_modifiers = {
            ShotType.DIRECT: self.config.get(
                "core.shot_analysis.shot_type_modifiers.direct", 1.0
            ),
            ShotType.BANK: self.config.get(
                "core.shot_analysis.shot_type_modifiers.bank", 1.3
            ),
            ShotType.COMBINATION: self.config.get(
                "core.shot_analysis.shot_type_modifiers.combination", 1.5
            ),
            ShotType.SAFETY: self.config.get(
                "core.shot_analysis.shot_type_modifiers.safety", 0.8
            ),
            ShotType.MASSE: self.config.get(
                "core.shot_analysis.shot_type_modifiers.masse", 1.8
            ),
            ShotType.BREAK: self.config.get(
                "core.shot_analysis.shot_type_modifiers.break", 0.6
            ),
        }

        final_difficulty = total_difficulty * type_modifiers.get(shot_type, 1.0)
        return min(final_difficulty, 1.0)  # Cap at 1.0

    def get_alternative_shots(
        self, game_state: GameState, max_alternatives: int = 3
    ) -> list[ShotAnalysis]:
        """Generate alternative shot suggestions."""
        alternatives = []

        # Get all possible target balls
        possible_targets = self._get_legal_target_balls(game_state)

        for target_id in possible_targets:
            target_ball = self._get_ball_by_id(game_state, target_id)
            if target_ball:
                try:
                    analysis = self.analyze_shot(game_state, target_id)
                    alternatives.append(analysis)
                except Exception:
                    continue  # Skip if analysis fails

        # Sort by success probability (descending) and difficulty (ascending)
        alternatives.sort(
            key=lambda x: (x.success_probability, -x.difficulty), reverse=True
        )

        return alternatives[:max_alternatives]

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

    def _determine_best_target(self, game_state: GameState) -> str:
        """Determine the best target ball based on game rules."""
        legal_targets = self._get_legal_target_balls(game_state)
        if not legal_targets:
            raise ValueError("No legal target balls found")

        # For now, return the first legal target
        # In a full implementation, this would consider strategy
        return legal_targets[0]

    def _get_legal_target_balls(self, game_state: GameState) -> list[str]:
        """Get list of legal target balls based on game type and rules."""
        legal_targets = []

        if game_state.game_type.value == "9ball":
            # In 9-ball, must hit lowest numbered ball first
            lowest_number = float("inf")
            for ball in game_state.balls:
                if (
                    not ball.is_cue_ball
                    and not ball.is_pocketed
                    and ball.number is not None
                    and ball.number < lowest_number
                ):
                    lowest_number = ball.number

            for ball in game_state.balls:
                if (
                    not ball.is_cue_ball
                    and not ball.is_pocketed
                    and ball.number == lowest_number
                ):
                    legal_targets.append(ball.id)

        elif game_state.game_type.value == "8ball":
            # In 8-ball, target depends on player's group
            # For simplicity, assume all non-8 balls are legal for now
            for ball in game_state.balls:
                if not ball.is_cue_ball and not ball.is_pocketed and ball.number != 8:
                    legal_targets.append(ball.id)

        else:
            # Practice mode - all balls are legal targets
            for ball in game_state.balls:
                if not ball.is_cue_ball and not ball.is_pocketed:
                    legal_targets.append(ball.id)

        return legal_targets

    def _calculate_optimal_aim_point(
        self, cue_ball: BallState, target_ball: BallState, table: TableState
    ) -> Vector2D:
        """Calculate optimal aiming point on target ball."""
        # Find closest pocket to target ball
        closest_pocket = min(
            table.pocket_positions,
            key=lambda p: self.geometry_utils.distance_between_points(
                (target_ball.position.x, target_ball.position.y), (p.x, p.y)
            ),
        )

        # Calculate ghost ball position (where cue ball center should be)
        # for straight shot to pocket
        dx = target_ball.position.x - closest_pocket.x
        dy = target_ball.position.y - closest_pocket.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance > 0:
            # Normalize and extend by ball diameter
            ball_diameter = target_ball.radius * 2
            ghost_x = target_ball.position.x + (dx / distance) * ball_diameter
            ghost_y = target_ball.position.y + (dy / distance) * ball_diameter

            # The aim point is on the target ball surface closest to ghost ball
            aim_dx = ghost_x - target_ball.position.x
            aim_dy = ghost_y - target_ball.position.y
            aim_distance = math.sqrt(aim_dx**2 + aim_dy**2)

            if aim_distance > 0:
                aim_x = (
                    target_ball.position.x
                    - (aim_dx / aim_distance) * target_ball.radius
                )
                aim_y = (
                    target_ball.position.y
                    - (aim_dy / aim_distance) * target_ball.radius
                )
                return Vector2D(aim_x, aim_y)

        # Fallback: center of target ball
        return Vector2D(target_ball.position.x, target_ball.position.y)

    def _calculate_recommended_force(
        self, cue_ball: BallState, target_ball: BallState, shot_type: ShotType
    ) -> float:
        """Calculate recommended force for shot."""
        # Base force on distance
        distance = self.geometry_utils.distance_between_points(
            (cue_ball.position.x, cue_ball.position.y),
            (target_ball.position.x, target_ball.position.y),
        )

        # Load force calculation parameters from config
        base = self.config.get("core.shot_analysis.force_calculation.base_force", 8.0)
        scale_factor = self.config.get(
            "core.shot_analysis.force_calculation.distance_scale_factor", 5.0
        )
        scale_divisor = self.config.get(
            "core.shot_analysis.force_calculation.distance_scale_divisor", 1000.0
        )

        base_force = base + (distance / scale_divisor) * scale_factor

        # Adjust for shot type using config
        type_multipliers = {
            ShotType.DIRECT: self.config.get(
                "core.shot_analysis.force_calculation.type_multipliers.direct", 1.0
            ),
            ShotType.BANK: self.config.get(
                "core.shot_analysis.force_calculation.type_multipliers.bank", 1.2
            ),
            ShotType.COMBINATION: self.config.get(
                "core.shot_analysis.force_calculation.type_multipliers.combination", 1.1
            ),
            ShotType.SAFETY: self.config.get(
                "core.shot_analysis.force_calculation.type_multipliers.safety", 0.7
            ),
            ShotType.MASSE: self.config.get(
                "core.shot_analysis.force_calculation.type_multipliers.masse", 0.8
            ),
            ShotType.BREAK: self.config.get(
                "core.shot_analysis.force_calculation.type_multipliers.break", 2.0
            ),
        }

        return base_force * type_multipliers.get(shot_type, 1.0)

    def _calculate_shot_angle(
        self, cue_position: Vector2D, aim_point: Vector2D
    ) -> float:
        """Calculate shot angle in degrees."""
        dx = aim_point.x - cue_position.x
        dy = aim_point.y - cue_position.y
        angle_rad = math.atan2(dy, dx)
        return math.degrees(angle_rad)

    def _calculate_success_probability(
        self, difficulty: float, shot_type: ShotType, game_state: GameState
    ) -> float:
        """Calculate probability of shot success."""
        # Base probability inversely related to difficulty
        base_probability = 1.0 - difficulty

        # Adjust for shot type reliability using config
        type_reliability = {
            ShotType.DIRECT: self.config.get(
                "core.shot_analysis.success_probability.type_reliability.direct", 1.0
            ),
            ShotType.BANK: self.config.get(
                "core.shot_analysis.success_probability.type_reliability.bank", 0.7
            ),
            ShotType.COMBINATION: self.config.get(
                "core.shot_analysis.success_probability.type_reliability.combination",
                0.6,
            ),
            ShotType.SAFETY: self.config.get(
                "core.shot_analysis.success_probability.type_reliability.safety", 0.9
            ),
            ShotType.MASSE: self.config.get(
                "core.shot_analysis.success_probability.type_reliability.masse", 0.4
            ),
            ShotType.BREAK: self.config.get(
                "core.shot_analysis.success_probability.type_reliability.break", 0.8
            ),
        }

        probability = base_probability * type_reliability.get(shot_type, 1.0)

        # Factor in table conditions using config
        friction_threshold = self.config.get(
            "core.shot_analysis.success_probability.high_friction_threshold", 0.3
        )
        friction_multiplier = self.config.get(
            "core.shot_analysis.success_probability.high_friction_multiplier", 0.9
        )

        if game_state.table.surface_friction > friction_threshold:
            probability *= friction_multiplier

        # Clamp using config values
        min_prob = self.config.get(
            "core.shot_analysis.success_probability.min_probability", 0.1
        )
        max_prob = self.config.get(
            "core.shot_analysis.success_probability.max_probability", 0.95
        )

        return max(min_prob, min(max_prob, probability))

    def _check_shot_legality(
        self, game_state: GameState, target_ball: BallState
    ) -> tuple[bool, list[IllegalShotReason]]:
        """Check if the shot is legal according to game rules."""
        reasons = []

        # Check if target ball is legal
        legal_targets = self._get_legal_target_balls(game_state)
        if target_ball.id not in legal_targets:
            if game_state.game_type.value == "9ball":
                reasons.append(IllegalShotReason.WRONG_BALL_FIRST)
            elif game_state.game_type.value == "8ball" and target_ball.number == 8:
                reasons.append(IllegalShotReason.EIGHT_BALL_EARLY)

        # Additional legality checks would go here
        # (cue ball contact, scratch prevention, etc.)

        return len(reasons) == 0, reasons

    def _identify_potential_problems(
        self,
        game_state: GameState,
        cue_ball: BallState,
        target_ball: BallState,
        shot_type: ShotType,
    ) -> list[str]:
        """Identify potential problems with the shot."""
        problems = []

        # Check for scratches
        if self._has_scratch_risk(game_state, cue_ball, target_ball):
            problems.append("High scratch risk")

        # Check for obstacles
        if self._is_direct_path_blocked(game_state, cue_ball, target_ball):
            problems.append("Direct path blocked")

        # Check for difficult angles using config threshold
        angle_difficulty = self._calculate_angle_difficulty(
            cue_ball, target_ball, game_state.table
        )
        angle_threshold = self.config.get(
            "core.shot_analysis.problem_thresholds.angle_difficulty_threshold", 0.7
        )
        if angle_difficulty > angle_threshold:
            problems.append("Very thin cut required")

        # Check for long distance using config threshold
        distance = self.geometry_utils.distance_between_points(
            (cue_ball.position.x, cue_ball.position.y),
            (target_ball.position.x, target_ball.position.y),
        )
        distance_threshold = self.config.get(
            "core.shot_analysis.problem_thresholds.long_distance_threshold", 1500
        )
        if distance > distance_threshold:
            problems.append("Long distance shot")

        return problems

    def _calculate_risk_factors(
        self,
        game_state: GameState,
        cue_ball: BallState,
        target_ball: BallState,
        shot_type: ShotType,
    ) -> dict[str, float]:
        """Calculate various risk factors for the shot."""
        risks = {}

        # Scratch risk using config values
        scratch_high = self.config.get(
            "core.shot_analysis.risk_factors.scratch_risk_high", 0.3
        )
        scratch_low = self.config.get(
            "core.shot_analysis.risk_factors.scratch_risk_low", 0.1
        )
        risks["scratch"] = (
            scratch_high
            if self._has_scratch_risk(game_state, cue_ball, target_ball)
            else scratch_low
        )

        # Miss risk based on difficulty
        difficulty = self.calculate_difficulty(
            game_state, cue_ball, target_ball, shot_type
        )
        risks["miss"] = difficulty

        # Leave opponent easy shot risk
        risks["opponent_runout"] = self._calculate_opponent_advantage_risk(
            game_state, target_ball
        )

        return risks

    def _predict_shot_outcome(
        self,
        game_state: GameState,
        cue_ball: BallState,
        target_ball: BallState,
        aim_point: Vector2D,
        force: float,
        shot_type: ShotType,
    ) -> str:
        """Predict the likely outcome of the shot."""
        success_prob = self._calculate_success_probability(
            self.calculate_difficulty(game_state, cue_ball, target_ball, shot_type),
            shot_type,
            game_state,
        )

        # Use config thresholds for outcome descriptions
        high_threshold = self.config.get(
            "core.shot_analysis.success_probability.high_threshold", 0.8
        )
        good_threshold = self.config.get(
            "core.shot_analysis.success_probability.good_threshold", 0.6
        )
        moderate_threshold = self.config.get(
            "core.shot_analysis.success_probability.moderate_threshold", 0.4
        )

        if success_prob > high_threshold:
            return "High probability of success"
        elif success_prob > good_threshold:
            return "Good chance of success"
        elif success_prob > moderate_threshold:
            return "Moderate chance of success"
        else:
            return "Low probability of success"

    # Helper methods for shot analysis
    def _is_direct_path_blocked(
        self, game_state: GameState, cue_ball: BallState, target_ball: BallState
    ) -> bool:
        """Check if direct path between cue ball and target is blocked."""
        # Simplified implementation - check for balls in the path
        for ball in game_state.balls:
            if ball.id == cue_ball.id or ball.id == target_ball.id or ball.is_pocketed:
                continue

            # Check if ball intersects with the line between cue and target
            if self._ball_intersects_line(
                cue_ball.position, target_ball.position, ball.position, ball.radius
            ):
                return True

        return False

    def _ball_intersects_line(
        self,
        line_start: Vector2D,
        line_end: Vector2D,
        ball_center: Vector2D,
        ball_radius: float,
    ) -> bool:
        """Check if a ball intersects with a line segment."""
        # Distance from ball center to line
        line_length = self.geometry_utils.distance_between_points(
            (line_start.x, line_start.y), (line_end.x, line_end.y)
        )

        if line_length == 0:
            return False

        # Vector from line start to ball center
        dx = ball_center.x - line_start.x
        dy = ball_center.y - line_start.y

        # Vector of the line
        line_dx = line_end.x - line_start.x
        line_dy = line_end.y - line_start.y

        # Project ball center onto line
        t = (dx * line_dx + dy * line_dy) / (line_length**2)
        t = max(0, min(1, t))  # Clamp to line segment

        # Closest point on line to ball center
        closest_x = line_start.x + t * line_dx
        closest_y = line_start.y + t * line_dy

        # Distance from ball center to closest point
        distance = self.geometry_utils.distance_between_points(
            (ball_center.x, ball_center.y), (closest_x, closest_y)
        )

        return distance < ball_radius

    def _is_cushion_shot(
        self, game_state: GameState, start_pos: Vector2D, aim_point: Vector2D
    ) -> bool:
        """Check if the shot is aimed at a cushion."""
        # Check if aim point is near table edges
        table = game_state.table
        margin = self.config.get(
            "core.shot_analysis.problem_thresholds.cushion_shot_margin", 50
        )

        return (
            aim_point.x < margin
            or aim_point.x > table.width - margin
            or aim_point.y < margin
            or aim_point.y > table.height - margin
        )

    def _is_combination_shot(
        self, game_state: GameState, cue_ball: BallState, target_ball: BallState
    ) -> bool:
        """Check if this is a combination shot."""
        # Simplified: check if there are balls between cue and target that could be hit first
        return self._is_direct_path_blocked(game_state, cue_ball, target_ball)

    def _is_safety_shot(self, game_state: GameState, target_ball: BallState) -> bool:
        """Determine if this is intended as a safety shot."""
        # This would typically be determined by player intention
        # For now, assume it's not a safety shot
        return False

    def _requires_masse(
        self, game_state: GameState, cue_ball: BallState, target_ball: BallState
    ) -> bool:
        """Check if shot requires masse (extreme curve)."""
        # Simplified: if direct path is severely blocked, might need masse
        return self._is_direct_path_blocked(
            game_state, cue_ball, target_ball
        ) and not self._is_combination_shot(game_state, cue_ball, target_ball)

    def _calculate_angle_difficulty(
        self, cue_ball: BallState, target_ball: BallState, table: TableState
    ) -> float:
        """Calculate difficulty based on cut angle."""
        # Find the angle between cue ball, target ball, and intended pocket
        closest_pocket = min(
            table.pocket_positions,
            key=lambda p: self.geometry_utils.distance_between_points(
                (target_ball.position.x, target_ball.position.y), (p.x, p.y)
            ),
        )

        # Calculate the cut angle
        # This is a simplified calculation - full implementation would be more complex
        cue_to_target = math.atan2(
            target_ball.position.y - cue_ball.position.y,
            target_ball.position.x - cue_ball.position.x,
        )

        target_to_pocket = math.atan2(
            closest_pocket.y - target_ball.position.y,
            closest_pocket.x - target_ball.position.x,
        )

        cut_angle = abs(cue_to_target - target_to_pocket)
        cut_angle = min(cut_angle, 2 * math.pi - cut_angle)  # Take smaller angle

        # Convert to difficulty (straight shot = 0, very thin cut = 1)
        return min(cut_angle / (math.pi / 2), 1.0)

    def _calculate_obstacle_difficulty(
        self, game_state: GameState, cue_ball: BallState, target_ball: BallState
    ) -> float:
        """Calculate difficulty based on interfering balls."""
        obstacles = 0
        radius_margin = self.config.get(
            "core.shot_analysis.difficulty_calculations.obstacles.radius_margin", 10
        )

        for ball in game_state.balls:
            if ball.id == cue_ball.id or ball.id == target_ball.id or ball.is_pocketed:
                continue

            if self._ball_intersects_line(
                cue_ball.position,
                target_ball.position,
                ball.position,
                ball.radius + radius_margin,
            ):
                obstacles += 1

        per_obstacle = self.config.get(
            "core.shot_analysis.difficulty_calculations.obstacles.per_obstacle_penalty",
            0.3,
        )
        max_diff = self.config.get(
            "core.shot_analysis.difficulty_calculations.obstacles.max_difficulty", 1.0
        )

        return min(obstacles * per_obstacle, max_diff)

    def _calculate_precision_difficulty(
        self, target_ball: BallState, table: TableState
    ) -> float:
        """Calculate difficulty based on precision required."""
        # Distance to closest pocket
        closest_distance = min(
            self.geometry_utils.distance_between_points(
                (target_ball.position.x, target_ball.position.y), (pocket.x, pocket.y)
            )
            for pocket in table.pocket_positions
        )

        # Normalize by pocket radius - closer pockets are easier
        multiplier = self.config.get(
            "core.shot_analysis.difficulty_calculations.precision.pocket_radius_multiplier",
            10,
        )
        return min(closest_distance / (table.pocket_radius * multiplier), 1.0)

    def _calculate_cushion_difficulty(self, shot_type: ShotType) -> float:
        """Calculate difficulty based on cushion bounces required."""
        if shot_type == ShotType.BANK:
            return self.config.get(
                "core.shot_analysis.difficulty_calculations.cushion.bank_shot", 0.7
            )
        elif shot_type == ShotType.MASSE:
            return self.config.get(
                "core.shot_analysis.difficulty_calculations.cushion.masse_shot", 0.5
            )
        else:
            return self.config.get(
                "core.shot_analysis.difficulty_calculations.cushion.other", 0.1
            )

    def _calculate_spin_difficulty(
        self, shot_type: ShotType, cue_ball: BallState, target_ball: BallState
    ) -> float:
        """Calculate difficulty based on spin required."""
        if shot_type == ShotType.MASSE:
            return self.config.get(
                "core.shot_analysis.difficulty_calculations.spin.masse_shot", 0.9
            )
        elif shot_type == ShotType.BANK:
            return self.config.get(
                "core.shot_analysis.difficulty_calculations.spin.bank_shot", 0.4
            )
        else:
            return self.config.get(
                "core.shot_analysis.difficulty_calculations.spin.other", 0.1
            )

    def _has_scratch_risk(
        self, game_state: GameState, cue_ball: BallState, target_ball: BallState
    ) -> bool:
        """Check if there's significant risk of scratching."""
        # Check if cue ball path after contact leads toward pockets
        # This is a simplified check
        risk_distance = self.config.get(
            "core.shot_analysis.problem_thresholds.scratch_risk_distance", 200
        )

        for pocket in game_state.table.pocket_positions:
            distance = self.geometry_utils.distance_between_points(
                (cue_ball.position.x, cue_ball.position.y), (pocket.x, pocket.y)
            )
            if distance < risk_distance:
                return True

        return False

    def _calculate_opponent_advantage_risk(
        self, game_state: GameState, target_ball: BallState
    ) -> float:
        """Calculate risk of leaving opponent with easy shots."""
        # Simplified calculation - in full implementation would analyze
        # resulting table layout after shot
        return self.config.get(
            "core.shot_analysis.risk_factors.opponent_advantage_default", 0.3
        )
