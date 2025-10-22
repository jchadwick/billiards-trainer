"""Player assistance feature algorithms.

All spatial calculations use the 4K coordinate system (3840×2160).
Safe zone radii and distance thresholds are scaled appropriately for 4K.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from config import Config, config

from ..constants_4k import (
    BALL_RADIUS_4K,
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    TABLE_HEIGHT_4K,
    TABLE_WIDTH_4K,
)
from ..coordinates import Vector2D
from ..models import BallState, GameState, GameType, ShotType, TableState
from ..utils.geometry import GeometryUtils
from .prediction import OutcomePredictor
from .shot import ShotAnalysis, ShotAnalyzer


class AssistanceLevel(Enum):
    """Assistance difficulty levels."""

    BEGINNER = "beginner"  # Maximum help, simple explanations
    INTERMEDIATE = "intermediate"  # Moderate help, some complexity
    ADVANCED = "advanced"  # Minimal help, complex strategies
    EXPERT = "expert"  # Analysis only, no hand-holding


class SafeZoneType(Enum):
    """Types of safe zones on the table."""

    DEFENSIVE = "defensive"  # Areas that make opponent's next shot difficult
    SCRATCH_SAFE = "scratch_safe"  # Areas that minimize scratch risk
    POSITION_PLAY = "position_play"  # Areas for good position on next shot
    CLUSTER_BREAK = "cluster_break"  # Areas to break up ball clusters


@dataclass
class AimingGuide:
    """Aiming assistance information."""

    target_point: Vector2D  # Exact point to aim at on target ball
    ghost_ball_position: Vector2D  # Where cue ball center should be
    cut_angle: float  # Cut angle in degrees
    confidence: float  # Confidence in this aiming point
    difficulty_level: str  # "easy", "medium", "hard"
    visual_aids: dict[str, Any]  # Data for visual overlays
    explanation: str  # Human-readable explanation


@dataclass
class PowerRecommendation:
    """Shot power level recommendations."""

    recommended_force: float  # In Newtons
    force_range: tuple[float, float]  # Min/max acceptable range
    power_level: str  # "soft", "medium", "hard"
    spin_recommendation: Optional[Vector2D]  # Recommended spin
    explanation: str  # Why this power level
    risk_assessment: dict[str, float]  # Various risk factors


@dataclass
class SafeZone:
    """Safe zone information."""

    zone_type: SafeZoneType
    center: Vector2D
    radius: float
    safety_score: float  # 0.0 (dangerous) to 1.0 (very safe)
    benefits: list[str]  # Why this zone is beneficial
    risks: list[str]  # Potential downsides
    access_difficulty: float  # How hard to reach this zone


@dataclass
class TargetRecommendation:
    """Target ball recommendation."""

    ball_id: str
    priority_score: float  # Higher = better target
    success_probability: float
    strategic_value: float  # Long-term game value
    reasons: list[str]  # Why this ball is recommended
    shot_analysis: ShotAnalysis  # Detailed analysis for this target
    alternative_approaches: list[str]  # Different ways to play this ball


@dataclass
class AssistancePackage:
    """Complete assistance information."""

    assistance_level: AssistanceLevel
    primary_recommendation: TargetRecommendation
    alternative_targets: list[TargetRecommendation]
    aiming_guide: AimingGuide
    power_recommendation: PowerRecommendation
    safe_zones: list[SafeZone]
    strategic_advice: str
    confidence: float

    # Adaptive learning data
    player_skill_estimate: dict[str, float]
    adjustment_factors: dict[str, float]

    # ML enhancement hooks
    ml_features: Optional[dict[str, Any]] = None
    ml_confidence: Optional[float] = None


class AssistanceEngine:
    """Player assistance features engine."""

    def __init__(self, cfg: Optional[Config] = None):
        """Initialize the assistance engine with configuration.

        Args:
            cfg: Optional configuration instance. If not provided, uses singleton.
        """
        self.shot_analyzer = ShotAnalyzer()
        self.outcome_predictor = OutcomePredictor()
        self.geometry_utils = GeometryUtils()

        # Configuration
        self.config = cfg if cfg is not None else config

        # Load assistance configuration
        assistance_config = self.config.get("core.assistance", {})

        # Assistance parameters by skill level
        level_configs = assistance_config.get("level_configs", {})
        self.assistance_configs = {
            AssistanceLevel.BEGINNER: level_configs.get("beginner", {}),
            AssistanceLevel.INTERMEDIATE: level_configs.get("intermediate", {}),
            AssistanceLevel.ADVANCED: level_configs.get("advanced", {}),
            AssistanceLevel.EXPERT: level_configs.get("expert", {}),
        }

        # Player skill tracking with defaults from config
        player_skills_config = assistance_config.get("player_skills", {})
        self.player_skills = {
            "aiming_accuracy": player_skills_config.get("default_aiming_accuracy", 0.5),
            "power_control": player_skills_config.get("default_power_control", 0.5),
            "strategic_thinking": player_skills_config.get(
                "default_strategic_thinking", 0.5
            ),
            "cue_ball_control": player_skills_config.get(
                "default_cue_ball_control", 0.5
            ),
            "pattern_recognition": player_skills_config.get(
                "default_pattern_recognition", 0.5
            ),
        }

    def provide_assistance(
        self,
        game_state: GameState,
        assistance_level: AssistanceLevel = AssistanceLevel.INTERMEDIATE,
        target_ball: Optional[str] = None,
    ) -> AssistancePackage:
        """Provide comprehensive assistance for the current shot."""
        config = self.assistance_configs[assistance_level]

        # Get target recommendations
        if target_ball:
            primary_target = self._analyze_specific_target(game_state, target_ball)
            alternatives = self._get_alternative_targets(
                game_state, exclude=[target_ball], max_count=config["max_alternatives"]
            )
        else:
            all_targets = self._get_all_target_recommendations(game_state)
            primary_target = all_targets[0] if all_targets else None
            alternatives = all_targets[1 : config["max_alternatives"] + 1]

        if not primary_target:
            raise ValueError("No valid targets found")

        # Generate aiming guide
        aiming_guide = self._generate_aiming_guide(
            game_state, primary_target, assistance_level
        )

        # Generate power recommendation
        power_recommendation = self._generate_power_recommendation(
            game_state, primary_target, assistance_level
        )

        # Calculate safe zones
        safe_zones = self._calculate_safe_zones(game_state, assistance_level)

        # Generate strategic advice
        strategic_advice = self._generate_strategic_advice(
            game_state, primary_target, assistance_level
        )

        # Calculate overall confidence
        confidence = self._calculate_assistance_confidence(
            primary_target, aiming_guide, power_recommendation
        )

        # Estimate player skill and adjustment factors
        player_skill_estimate = self._estimate_player_skills(game_state)
        adjustment_factors = self._calculate_adjustment_factors(
            player_skill_estimate, assistance_level
        )

        return AssistancePackage(
            assistance_level=assistance_level,
            primary_recommendation=primary_target,
            alternative_targets=alternatives,
            aiming_guide=aiming_guide,
            power_recommendation=power_recommendation,
            safe_zones=safe_zones,
            strategic_advice=strategic_advice,
            confidence=confidence,
            player_skill_estimate=player_skill_estimate,
            adjustment_factors=adjustment_factors,
        )

    def suggest_optimal_aim_point(
        self, game_state: GameState, target_ball_id: str
    ) -> Vector2D:
        """Suggest optimal aiming point on target ball."""
        target_ball = self._get_ball_by_id(game_state, target_ball_id)
        cue_ball = self._get_cue_ball(game_state)

        if not target_ball or not cue_ball:
            raise ValueError("Missing target ball or cue ball")

        # Find the best pocket for this target ball
        best_pocket = self._find_best_pocket(target_ball, game_state.table)

        # Calculate ghost ball position (where cue ball center should be)
        ghost_ball_pos = self._calculate_ghost_ball_position(
            target_ball, best_pocket, target_ball.radius * 2
        )

        # Calculate aim point (contact point on target ball)
        direction = Vector2D.from_4k(
            ghost_ball_pos.x - target_ball.position.x,
            ghost_ball_pos.y - target_ball.position.y,
        ).normalize()

        aim_point = Vector2D.from_4k(
            target_ball.position.x - direction.x * target_ball.radius,
            target_ball.position.y - direction.y * target_ball.radius,
        )

        return aim_point

    def recommend_shot_power(
        self, game_state: GameState, shot_analysis: ShotAnalysis
    ) -> PowerRecommendation:
        """Recommend optimal shot power level."""
        # Base power from shot analysis
        base_force = shot_analysis.recommended_force

        # Get power calculation config
        power_config = self.config.get("core.assistance.power_calculation", {})
        type_adjustments = power_config.get("shot_type_adjustments", {})

        # Adjust based on shot type
        shot_type_key = (
            shot_analysis.shot_type.value
            if hasattr(shot_analysis.shot_type, "value")
            else str(shot_analysis.shot_type).lower()
        )
        adjustment = type_adjustments.get(shot_type_key, 1.0)
        adjusted_force = base_force * adjustment

        # Calculate acceptable range (±tolerance%)
        tolerance = power_config.get("force_range_tolerance", 0.15)
        force_range = (
            adjusted_force * (1 - tolerance),
            adjusted_force * (1 + tolerance),
        )

        # Determine power level category
        soft_threshold = power_config.get("soft_power_threshold", 8.0)
        medium_threshold = power_config.get("medium_power_threshold", 12.0)

        if adjusted_force < soft_threshold:
            power_level = "soft"
        elif adjusted_force < medium_threshold:
            power_level = "medium"
        else:
            power_level = "hard"

        # Spin recommendation
        spin_recommendation = self._calculate_optimal_spin(game_state, shot_analysis)

        # Risk assessment
        risk_config = self.config.get("core.assistance.risk_assessment", {})
        risk_assessment = {
            "scratch_risk": shot_analysis.risk_factors.get(
                "scratch", risk_config.get("default_scratch_risk", 0.1)
            ),
            "miss_risk": shot_analysis.risk_factors.get(
                "miss", shot_analysis.difficulty
            ),
            "position_risk": shot_analysis.risk_factors.get(
                "opponent_runout", risk_config.get("default_opponent_runout_risk", 0.3)
            ),
        }

        # Generate explanation
        explanation = self._generate_power_explanation(
            adjusted_force, power_level, shot_analysis
        )

        return PowerRecommendation(
            recommended_force=adjusted_force,
            force_range=force_range,
            power_level=power_level,
            spin_recommendation=spin_recommendation,
            explanation=explanation,
            risk_assessment=risk_assessment,
        )

    def identify_best_target(self, game_state: GameState) -> str:
        """Identify the best target ball for current situation."""
        recommendations = self._get_all_target_recommendations(game_state)

        if not recommendations:
            raise ValueError("No legal target balls found")

        return recommendations[0].ball_id

    def calculate_safe_zones(self, game_state: GameState) -> list[SafeZone]:
        """Calculate safe zones for cue ball placement."""
        safe_zones = []

        # Defensive safe zones (make opponent's shot difficult)
        defensive_zones = self._find_defensive_zones(game_state)
        safe_zones.extend(defensive_zones)

        # Scratch-safe zones (minimize scratch risk)
        scratch_safe_zones = self._find_scratch_safe_zones(game_state)
        safe_zones.extend(scratch_safe_zones)

        # Position play zones (good position for next shot)
        position_zones = self._find_position_play_zones(game_state)
        safe_zones.extend(position_zones)

        # Cluster breaking zones
        cluster_zones = self._find_cluster_break_zones(game_state)
        safe_zones.extend(cluster_zones)

        # Sort by safety score
        safe_zones.sort(key=lambda z: z.safety_score, reverse=True)

        return safe_zones

    def adjust_for_difficulty(
        self, assistance: AssistancePackage, player_skill: dict[str, float]
    ) -> AssistancePackage:
        """Adjust assistance based on player skill level."""
        # Get skill thresholds and visual aid configs
        skill_thresholds = self.config.get("core.assistance.skill_thresholds", {})
        visual_config = self.config.get("core.assistance.visual_aids", {})
        power_config = self.config.get("core.assistance.power_calculation", {})

        aiming_low = skill_thresholds.get("aiming_accuracy_low", 0.3)
        aiming_high = skill_thresholds.get("aiming_accuracy_high", 0.7)
        power_low = skill_thresholds.get("power_control_low", 0.3)
        strategic_low = skill_thresholds.get("strategic_thinking_low", 0.3)
        strategic_high = skill_thresholds.get("strategic_thinking_high", 0.7)

        # Adjust aiming guide based on aiming accuracy
        if player_skill.get("aiming_accuracy", 0.5) < aiming_low:
            # Beginner aiming - make ghost ball more prominent
            assistance.aiming_guide.visual_aids["ghost_ball_opacity"] = (
                visual_config.get("beginner_ghost_ball_opacity", 0.8)
            )
            assistance.aiming_guide.visual_aids["aim_line_width"] = visual_config.get(
                "beginner_aim_line_width", "thick"
            )
        elif player_skill.get("aiming_accuracy", 0.5) > aiming_high:
            # Advanced aiming - minimal visual aids
            assistance.aiming_guide.visual_aids["ghost_ball_opacity"] = (
                visual_config.get("advanced_ghost_ball_opacity", 0.3)
            )
            assistance.aiming_guide.visual_aids["aim_line_width"] = visual_config.get(
                "advanced_aim_line_width", "thin"
            )

        # Adjust power recommendation based on power control
        if player_skill.get("power_control", 0.5) < power_low:
            # Suggest more conservative power
            conservative_multiplier = power_config.get(
                "conservative_power_multiplier", 0.9
            )
            assistance.power_recommendation.recommended_force *= conservative_multiplier
            assistance.power_recommendation.explanation += (
                " (Conservative power for better control)"
            )

        # Adjust strategic advice based on strategic thinking
        if player_skill.get("strategic_thinking", 0.5) < strategic_low:
            assistance.strategic_advice = (
                f"Focus on making this ball. {assistance.strategic_advice}"
            )
        elif player_skill.get("strategic_thinking", 0.5) > strategic_high:
            assistance.strategic_advice = f"{assistance.strategic_advice} Consider the pattern for your next 2-3 shots."

        return assistance

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

    def _get_all_target_recommendations(
        self, game_state: GameState
    ) -> list[TargetRecommendation]:
        """Get all possible target recommendations ranked by priority."""
        recommendations = []

        # Get legal target balls
        legal_targets = self._get_legal_targets(game_state)

        for ball_id in legal_targets:
            try:
                recommendation = self._analyze_specific_target(game_state, ball_id)
                recommendations.append(recommendation)
            except Exception:
                continue  # Skip if analysis fails

        # Sort by priority score (higher is better)
        recommendations.sort(key=lambda r: r.priority_score, reverse=True)

        return recommendations

    def _analyze_specific_target(
        self, game_state: GameState, ball_id: str
    ) -> TargetRecommendation:
        """Analyze a specific target ball and create recommendation."""
        shot_analysis = self.shot_analyzer.analyze_shot(game_state, ball_id)

        # Calculate priority score based on multiple factors
        priority_score = self._calculate_target_priority(
            game_state, ball_id, shot_analysis
        )

        # Calculate strategic value
        strategic_value = self._calculate_strategic_value(
            game_state, ball_id, shot_analysis
        )

        # Generate reasons for this recommendation
        reasons = self._generate_target_reasons(shot_analysis, strategic_value)

        # Generate alternative approaches
        alternatives = self._generate_alternative_approaches(game_state, ball_id)

        return TargetRecommendation(
            ball_id=ball_id,
            priority_score=priority_score,
            success_probability=shot_analysis.success_probability,
            strategic_value=strategic_value,
            reasons=reasons,
            shot_analysis=shot_analysis,
            alternative_approaches=alternatives,
        )

    def _get_alternative_targets(
        self, game_state: GameState, exclude: list[str], max_count: int
    ) -> list[TargetRecommendation]:
        """Get alternative target recommendations."""
        all_recommendations = self._get_all_target_recommendations(game_state)

        # Filter out excluded targets
        alternatives = [r for r in all_recommendations if r.ball_id not in exclude]

        return alternatives[:max_count]

    def _generate_aiming_guide(
        self,
        game_state: GameState,
        target: TargetRecommendation,
        assistance_level: AssistanceLevel,
    ) -> AimingGuide:
        """Generate comprehensive aiming guidance."""
        config = self.assistance_configs[assistance_level]

        target_ball = self._get_ball_by_id(game_state, target.ball_id)
        cue_ball = self._get_cue_ball(game_state)

        # Calculate aiming information
        target_point = target.shot_analysis.recommended_aim_point
        ghost_ball_position = self._calculate_ghost_ball_position(
            target_ball, target_point, target_ball.radius * 2
        )

        # Calculate cut angle
        cut_angle = self._calculate_cut_angle(cue_ball, target_ball, target_point)

        # Determine difficulty level
        difficulty_thresholds = self.config.get("core.assistance.skill_thresholds", {})
        diff_easy = difficulty_thresholds.get("difficulty_easy", 0.3)
        diff_medium = difficulty_thresholds.get("difficulty_medium", 0.6)

        if target.shot_analysis.difficulty < diff_easy:
            difficulty_level = "easy"
        elif target.shot_analysis.difficulty < diff_medium:
            difficulty_level = "medium"
        else:
            difficulty_level = "hard"

        # Generate visual aids configuration
        visual_config = self.config.get("core.assistance.visual_aids", {})
        visual_aids = {
            "show_ghost_ball": config["show_ghost_ball"],
            "show_aim_line": config["show_aim_line"],
            "ghost_ball_opacity": visual_config.get("default_ghost_ball_opacity", 0.6),
            "aim_line_width": visual_config.get("default_aim_line_width", "medium"),
            "contact_point_highlight": visual_config.get(
                "contact_point_highlight", True
            ),
        }

        # Generate explanation
        explanation = self._generate_aiming_explanation(
            cut_angle, difficulty_level, config["explanation_detail"]
        )

        return AimingGuide(
            target_point=target_point,
            ghost_ball_position=ghost_ball_position,
            cut_angle=cut_angle,
            confidence=target.shot_analysis.success_probability,
            difficulty_level=difficulty_level,
            visual_aids=visual_aids,
            explanation=explanation,
        )

    def _generate_power_recommendation(
        self,
        game_state: GameState,
        target: TargetRecommendation,
        assistance_level: AssistanceLevel,
    ) -> PowerRecommendation:
        """Generate power recommendation for the shot."""
        return self.recommend_shot_power(game_state, target.shot_analysis)

    def _calculate_safe_zones(
        self, game_state: GameState, assistance_level: AssistanceLevel
    ) -> list[SafeZone]:
        """Calculate safe zones based on assistance level."""
        config = self.assistance_configs[assistance_level]

        if not config["show_safe_zones"]:
            return []

        return self.calculate_safe_zones(game_state)

    def _generate_strategic_advice(
        self,
        game_state: GameState,
        target: TargetRecommendation,
        assistance_level: AssistanceLevel,
    ) -> str:
        """Generate strategic advice for the current situation."""
        config = self.assistance_configs[assistance_level]
        detail_level = config["explanation_detail"]

        advice_parts = []

        # Get thresholds from config
        thresholds = self.config.get("core.assistance.skill_thresholds", {})
        diff_easy = thresholds.get("difficulty_easy", 0.3)
        diff_hard = thresholds.get("difficulty_hard", 0.7)
        thresholds.get("strategic_value_high", 0.6)

        # Basic shot advice
        if target.shot_analysis.difficulty < diff_easy:
            advice_parts.append("This is a relatively easy shot.")
        elif target.shot_analysis.difficulty > diff_hard:
            advice_parts.append("This is a challenging shot - take your time.")

        # Add problems if any
        if target.shot_analysis.potential_problems:
            if detail_level == "detailed":
                problems = ", ".join(target.shot_analysis.potential_problems)
                advice_parts.append(f"Watch out for: {problems}")
            elif detail_level == "moderate":
                advice_parts.append("Be aware of the identified shot challenges.")

        # Add strategic considerations
        # Note: Using strategic_value_very_high is intentional - it's slightly higher than
        # strategic_value_high to ensure only really strategic balls get this advice
        strategic_very_high = thresholds.get("strategic_value_very_high", 0.7)
        if target.strategic_value > strategic_very_high and detail_level in [
            "detailed",
            "moderate",
        ]:
            advice_parts.append("This ball has good strategic value for your position.")

        # Position play advice
        if game_state.game_type != GameType.PRACTICE and detail_level == "detailed":
            advice_parts.append("Consider your position for the next shot.")

        return " ".join(advice_parts)

    def _calculate_assistance_confidence(
        self,
        target: TargetRecommendation,
        aiming_guide: AimingGuide,
        power_recommendation: PowerRecommendation,
    ) -> float:
        """Calculate overall confidence in the assistance provided."""
        factors = [
            target.success_probability,
            aiming_guide.confidence,
            1.0
            - sum(power_recommendation.risk_assessment.values())
            / len(power_recommendation.risk_assessment),
        ]

        return sum(factors) / len(factors)

    def _estimate_player_skills(self, game_state: GameState) -> dict[str, float]:
        """Estimate player skill levels based on game history."""
        # This would analyze game history to estimate skills
        # For now, return default values
        return dict(self.player_skills)

    def _calculate_adjustment_factors(
        self, player_skills: dict[str, float], assistance_level: AssistanceLevel
    ) -> dict[str, float]:
        """Calculate adjustment factors based on player skills and assistance level."""
        # Get adjustment factors from config
        adjustment_config = self.config.get("core.assistance.adjustment_factors", {})

        base_factors = adjustment_config.get(
            "base_factors",
            {
                "aiming_assistance_strength": 1.0,
                "power_assistance_strength": 1.0,
                "strategic_guidance_depth": 1.0,
                "visual_aid_prominence": 1.0,
            },
        )

        # Adjust based on assistance level
        level_multipliers_config = adjustment_config.get("level_multipliers", {})
        level_multipliers = {
            AssistanceLevel.BEGINNER: level_multipliers_config.get("beginner", 1.5),
            AssistanceLevel.INTERMEDIATE: level_multipliers_config.get(
                "intermediate", 1.0
            ),
            AssistanceLevel.ADVANCED: level_multipliers_config.get("advanced", 0.7),
            AssistanceLevel.EXPERT: level_multipliers_config.get("expert", 0.3),
        }

        multiplier = level_multipliers[assistance_level]

        return {k: v * multiplier for k, v in base_factors.items()}

    def _get_legal_targets(self, game_state: GameState) -> list[str]:
        """Get list of legal target balls."""
        legal_targets = []

        if game_state.game_type == GameType.NINE_BALL:
            # Must hit lowest numbered ball first
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

        elif game_state.game_type == GameType.EIGHT_BALL:
            # Simplified - all non-8 balls are legal
            for ball in game_state.balls:
                if not ball.is_cue_ball and not ball.is_pocketed and ball.number != 8:
                    legal_targets.append(ball.id)

        else:
            # Practice mode - all balls are legal
            for ball in game_state.balls:
                if not ball.is_cue_ball and not ball.is_pocketed:
                    legal_targets.append(ball.id)

        return legal_targets

    def _calculate_target_priority(
        self, game_state: GameState, ball_id: str, shot_analysis: ShotAnalysis
    ) -> float:
        """Calculate priority score for a target ball."""
        priority = 0.0

        # Get priority weights from config
        weights = self.config.get("core.assistance.priority_weights", {})
        success_weight = weights.get("success_probability", 0.4)
        difficulty_weight = weights.get("shot_difficulty", 0.3)
        strategic_weight = weights.get("strategic_value", 0.3)

        # Success probability is major factor
        priority += shot_analysis.success_probability * success_weight

        # Ease of shot (inverse of difficulty)
        priority += (1.0 - shot_analysis.difficulty) * difficulty_weight

        # Strategic value based on game type
        strategic_value = self._calculate_strategic_value(
            game_state, ball_id, shot_analysis
        )
        priority += strategic_value * strategic_weight

        return min(priority, 1.0)

    def _calculate_strategic_value(
        self, game_state: GameState, ball_id: str, shot_analysis: ShotAnalysis
    ) -> float:
        """Calculate strategic value of targeting this ball."""
        # Get strategic value config
        strategic_config = self.config.get("core.assistance.strategic_value", {})

        # Base strategic value
        value = strategic_config.get("base_value", 0.5)

        # In 9-ball, lower numbered balls have higher value
        if game_state.game_type == GameType.NINE_BALL:
            ball = self._get_ball_by_id(game_state, ball_id)
            if ball and ball.number:
                # Lower numbers are more valuable
                number_weight = strategic_config.get("nine_ball_number_weight", 0.3)
                value += (10 - ball.number) / 10 * number_weight

        # Consider position for next shot
        # (This would be more sophisticated in a full implementation)
        if shot_analysis.shot_type == ShotType.DIRECT:
            direct_bonus = strategic_config.get("direct_shot_bonus", 0.2)
            value += direct_bonus  # Direct shots often leave better position

        return min(value, 1.0)

    def _generate_target_reasons(
        self, shot_analysis: ShotAnalysis, strategic_value: float
    ) -> list[str]:
        """Generate reasons why this target is recommended."""
        reasons = []

        # Get skill thresholds from config
        thresholds = self.config.get("core.assistance.skill_thresholds", {})
        prob_high = thresholds.get("success_probability_high", 0.7)
        strategic_high = thresholds.get("strategic_value_high", 0.6)
        diff_easy_max = thresholds.get("difficulty_easy_max", 0.4)

        if shot_analysis.success_probability > prob_high:
            reasons.append("High probability of success")

        if shot_analysis.difficulty < diff_easy_max:
            reasons.append("Relatively easy shot")

        if strategic_value > strategic_high:
            reasons.append("Good strategic value")

        if shot_analysis.shot_type == ShotType.DIRECT:
            reasons.append("Direct shot with clean contact")

        if not shot_analysis.potential_problems:
            reasons.append("No major obstacles identified")

        return reasons

    def _generate_alternative_approaches(
        self, game_state: GameState, ball_id: str
    ) -> list[str]:
        """Generate alternative approaches for playing this ball."""
        approaches = []

        # This would analyze different ways to approach the shot
        approaches.append("Direct shot to closest pocket")

        # Check if bank shot is possible
        target_ball = self._get_ball_by_id(game_state, ball_id)
        if target_ball:
            # Simplified bank shot check
            approaches.append("Bank shot option")

        # Safety play option
        approaches.append("Defensive safety play")

        return approaches

    def _find_best_pocket(self, target_ball: BallState, table: TableState) -> Vector2D:
        """Find the best pocket for the target ball."""
        return min(
            table.pocket_positions,
            key=lambda p: self.geometry_utils.distance_between_vectors(
                target_ball.position, p
            ),
        )

    def _calculate_ghost_ball_position(
        self, target_ball: BallState, target_point: Vector2D, ball_diameter: float
    ) -> Vector2D:
        """Calculate where cue ball center should be (ghost ball position)."""
        direction = Vector2D(
            target_ball.position.x - target_point.x,
            target_ball.position.y - target_point.y,
            target_ball.position.scale,
        ).normalize()

        return Vector2D(
            target_ball.position.x + direction.x * ball_diameter,
            target_ball.position.y + direction.y * ball_diameter,
            target_ball.position.scale,
        )

    def _calculate_cut_angle(
        self, cue_ball: BallState, target_ball: BallState, target_point: Vector2D
    ) -> float:
        """Calculate the cut angle for the shot."""
        # Vector from cue ball to target ball
        cue_to_target = Vector2D.from_4k(
            target_ball.position.x - cue_ball.position.x,
            target_ball.position.y - cue_ball.position.y,
        )

        # Vector from target ball to target point
        target_to_point = Vector2D.from_4k(
            target_point.x - target_ball.position.x,
            target_point.y - target_ball.position.y,
        )

        # Calculate angle between vectors
        angle = self.geometry_utils.angle_between_vectors(
            (cue_to_target.x, cue_to_target.y), (target_to_point.x, target_to_point.y)
        )

        return math.degrees(angle)

    def _calculate_optimal_spin(
        self, game_state: GameState, shot_analysis: ShotAnalysis
    ) -> Optional[Vector2D]:
        """Calculate optimal spin for the shot."""
        # Get spin recommendations from config
        spin_config = self.config.get("core.assistance.spin_recommendations", {})

        # Simplified spin calculation (spin values are unitless, use 4K scale)
        SCALE_4K = (1.0, 1.0)
        if shot_analysis.shot_type == ShotType.BANK:
            # Bank shots often benefit from slight running english
            return Vector2D(
                spin_config.get("bank_shot_english_x", 0.3),
                spin_config.get("bank_shot_english_y", 0.0),
                SCALE_4K,
            )
        elif shot_analysis.shot_type == ShotType.SAFETY:
            # Safety shots often use draw to control cue ball
            return Vector2D(
                spin_config.get("safety_draw_x", 0.0),
                spin_config.get("safety_draw_y", -0.2),
                SCALE_4K,
            )
        else:
            # Most shots are fine with center ball
            return Vector2D(
                spin_config.get("center_ball_x", 0.0),
                spin_config.get("center_ball_y", 0.0),
                SCALE_4K,
            )

    def _generate_power_explanation(
        self, force: float, power_level: str, shot_analysis: ShotAnalysis
    ) -> str:
        """Generate explanation for power recommendation."""
        explanations = {
            "soft": f"Use soft power ({power_level}) for better control",
            "medium": f"Use medium power ({power_level}) for good balance of control and effectiveness",
            "hard": f"Use firm power ({power_level}) to ensure the ball reaches the pocket",
        }

        base_explanation = explanations.get(
            power_level, "Use appropriate power for the shot"
        )

        # Add shot-specific advice
        if shot_analysis.shot_type == ShotType.BANK:
            base_explanation += ". Bank shots need extra power for cushion contact."
        elif shot_analysis.shot_type == ShotType.SAFETY:
            base_explanation += ". Safety shots require precise speed control."

        return base_explanation

    def _generate_aiming_explanation(
        self, cut_angle: float, difficulty: str, detail_level: str
    ) -> str:
        """Generate explanation for aiming guidance."""
        if detail_level == "analysis_only":
            return f"Cut angle: {cut_angle:.1f}°"

        # Get cut angle thresholds from config
        angle_config = self.config.get("core.assistance.cut_angle_descriptions", {})
        straight_max = angle_config.get("straight_angle_max", 15.0)
        slight_max = angle_config.get("slight_cut_max", 30.0)
        moderate_max = angle_config.get("moderate_cut_max", 60.0)

        if cut_angle < straight_max:
            angle_desc = "nearly straight"
        elif cut_angle < slight_max:
            angle_desc = "slight cut"
        elif cut_angle < moderate_max:
            angle_desc = "moderate cut"
        else:
            angle_desc = "thin cut"

        if detail_level == "detailed":
            return f"This is a {angle_desc} shot with a {cut_angle:.1f}° cut angle. {difficulty.capitalize()} difficulty level."
        elif detail_level == "moderate":
            return f"{angle_desc.capitalize()} shot, {difficulty} difficulty"
        else:
            return f"{angle_desc.capitalize()} shot"

    # Safe zone calculation methods
    def _find_defensive_zones(self, game_state: GameState) -> list[SafeZone]:
        """Find zones that make opponent's next shot difficult (4K pixels)."""
        zones = []

        # Get defensive zone config (4K pixel values)
        defensive_config = self.config.get("core.assistance.safe_zones.defensive", {})
        # Cluster distance: reasonable clustering distance in 4K pixels (~480 pixels = ~15% of table width)
        cluster_distance = defensive_config.get("cluster_distance_threshold", 480)
        min_nearby = defensive_config.get("min_nearby_balls", 2)
        # Zone radius in 4K pixels (~320 pixels = ~10% of table width)
        zone_radius = defensive_config.get("zone_radius", 320)
        safety_score = defensive_config.get("safety_score", 0.7)
        access_difficulty = defensive_config.get("access_difficulty", 0.6)

        # Simplified: areas behind clusters of balls
        for _i, ball in enumerate(game_state.balls):
            if ball.is_cue_ball or ball.is_pocketed:
                continue

            # Check if there are other balls nearby (cluster)
            nearby_balls = [
                other
                for other in game_state.balls
                if (
                    not other.is_cue_ball
                    and not other.is_pocketed
                    and other.id != ball.id
                    and ball.distance_to(other) < cluster_distance
                )
            ]

            if len(nearby_balls) >= min_nearby:
                # This is a cluster - zone behind it is defensive
                zone = SafeZone(
                    zone_type=SafeZoneType.DEFENSIVE,
                    center=Vector2D.from_4k(ball.position.x, ball.position.y),
                    radius=zone_radius,
                    safety_score=safety_score,
                    benefits=["Makes opponent's shots more difficult"],
                    risks=["May be hard to escape from"],
                    access_difficulty=access_difficulty,
                )
                zones.append(zone)

        return zones

    def _find_scratch_safe_zones(self, game_state: GameState) -> list[SafeZone]:
        """Find zones that minimize scratch risk (4K pixels)."""
        zones = []

        # Get scratch-safe zone config (4K pixel values)
        scratch_config = self.config.get("core.assistance.safe_zones.scratch_safe", {})
        # Safe distance from pocket in 4K pixels (~640 pixels = ~20% of table width)
        safe_distance = scratch_config.get("safe_distance_from_pocket", 640)
        # Zone radius in 4K pixels (~256 pixels = ~8% of table width)
        zone_radius = scratch_config.get("zone_radius", 256)
        safety_score = scratch_config.get("safety_score", 0.8)
        access_difficulty = scratch_config.get("access_difficulty", 0.4)

        # Areas away from pockets
        for pocket in game_state.table.pocket_positions:
            # Find direction away from pocket center
            table_center = Vector2D.from_4k(
                game_state.table.width / 2, game_state.table.height / 2
            )
            away_direction = Vector2D.from_4k(
                table_center.x - pocket.x, table_center.y - pocket.y
            ).normalize()

            safe_center = Vector2D.from_4k(
                pocket.x + away_direction.x * safe_distance,
                pocket.y + away_direction.y * safe_distance,
            )

            # Check if position is on table
            if game_state.table.is_point_on_table(safe_center):
                zone = SafeZone(
                    zone_type=SafeZoneType.SCRATCH_SAFE,
                    center=safe_center,
                    radius=zone_radius,
                    safety_score=safety_score,
                    benefits=["Low scratch risk"],
                    risks=["May limit shot options"],
                    access_difficulty=access_difficulty,
                )
                zones.append(zone)

        return zones

    def _find_position_play_zones(self, game_state: GameState) -> list[SafeZone]:
        """Find zones good for position on next shot (4K pixels)."""
        zones = []

        # Get position play zone config
        position_config = self.config.get(
            "core.assistance.safe_zones.position_play", {}
        )
        x_factor = position_config.get("table_position_x_factor", 0.4)
        y_factor = position_config.get("table_position_y_factor", 0.5)
        # Zone radius in 4K pixels (~480 pixels = ~15% of table width)
        zone_radius = position_config.get("zone_radius", 480)
        safety_score = position_config.get("safety_score", 0.6)
        access_difficulty = position_config.get("access_difficulty", 0.5)
        min_legal = position_config.get("min_legal_targets", 2)

        # Simplified: areas with good angles to multiple target balls
        legal_targets = self._get_legal_targets(game_state)

        if len(legal_targets) >= min_legal:
            # Find areas with good sight lines to multiple targets
            zone = SafeZone(
                zone_type=SafeZoneType.POSITION_PLAY,
                center=Vector2D(
                    game_state.table.width * x_factor,
                    game_state.table.height * y_factor,
                ),
                radius=zone_radius,
                safety_score=safety_score,
                benefits=["Good position for next shot"],
                risks=["May require precise cue ball control"],
                access_difficulty=access_difficulty,
            )
            zones.append(zone)

        return zones

    def _find_cluster_break_zones(self, game_state: GameState) -> list[SafeZone]:
        """Find zones for breaking up ball clusters."""
        zones = []

        # Find clusters and identify good break-up positions
        # (Simplified implementation)

        return zones  # Return empty for now
