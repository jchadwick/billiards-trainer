"""Basic shot analysis algorithms.

All spatial calculations use the 4K coordinate system (3840×2160).

Simplified to basic shot type identification only:
- No difficulty calculations
- No success probability
- No risk analysis
- No alternative shot suggestions
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config import Config, config

from ..models import BallState, GameState, ShotType
from ..utils.geometry import GeometryUtils


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
    """Basic shot analysis - type identification only."""

    shot_type: ShotType
    target_ball_id: Optional[str] = None
    is_legal: bool = True
    illegal_reasons: list[IllegalShotReason] = None

    def __post_init__(self):
        if self.illegal_reasons is None:
            self.illegal_reasons = []


class ShotAnalyzer:
    """Basic shot analysis engine."""

    def __init__(self, cfg: Optional[Config] = None):
        self.geometry_utils = GeometryUtils()
        self.config = cfg if cfg is not None else config

    def analyze_shot(
        self,
        game_state: GameState,
        target_ball: Optional[str] = None,
    ) -> ShotAnalysis:
        """Analyze current shot setup - identify basic shot type only.

        Args:
            game_state: Current game state
            target_ball: Optional target ball ID

        Returns:
            Basic shot analysis with type identification
        """
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
        shot_type = self._identify_shot_type(game_state, cue_ball, target_ball_obj)

        # Check legality (basic)
        is_legal, illegal_reasons = self._check_shot_legality(
            game_state, target_ball_obj
        )

        return ShotAnalysis(
            shot_type=shot_type,
            target_ball_id=target_ball,
            is_legal=is_legal,
            illegal_reasons=illegal_reasons,
        )

    def _identify_shot_type(
        self,
        game_state: GameState,
        cue_ball: BallState,
        target_ball: BallState,
    ) -> ShotType:
        """Identify the type of shot being attempted."""
        # Check if it's a break shot
        if game_state.is_break:
            return ShotType.BREAK

        # Check if direct path is blocked
        path_blocked = self._is_direct_path_blocked(game_state, cue_ball, target_ball)

        # If path blocked, likely a bank or combination shot
        if path_blocked:
            # For now, call it combination
            # Could enhance later to detect bank vs combination
            return ShotType.COMBINATION

        # Default to direct shot
        return ShotType.DIRECT

    def _check_shot_legality(
        self, game_state: GameState, target_ball: BallState
    ) -> tuple[bool, list[IllegalShotReason]]:
        """Basic legality check."""
        # For now, just return legal
        # Can enhance later with actual rule checking
        return True, []

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
        """Determine best target ball - for now just return first non-cue ball."""
        for ball in game_state.balls:
            if not ball.is_cue_ball and not ball.is_pocketed:
                return ball.id
        raise ValueError("No target balls available")

    def _is_direct_path_blocked(
        self, game_state: GameState, cue_ball: BallState, target_ball: BallState
    ) -> bool:
        """Check if direct path from cue ball to target is blocked."""
        # Get line from cue ball to target
        line_start = (cue_ball.position.x, cue_ball.position.y)
        line_end = (target_ball.position.x, target_ball.position.y)

        # Check each ball (except cue and target) for intersection
        for ball in game_state.balls:
            if ball.id == cue_ball.id or ball.id == target_ball.id or ball.is_pocketed:
                continue

            # Check if ball intersects the line
            if self._ball_intersects_line(ball, line_start, line_end):
                return True

        return False

    def _ball_intersects_line(
        self,
        ball: BallState,
        line_start: tuple[float, float],
        line_end: tuple[float, float],
    ) -> bool:
        """Check if a ball intersects a line segment."""
        ball_center = (ball.position.x, ball.position.y)

        # Calculate perpendicular distance from ball center to line
        distance = self.geometry_utils.perpendicular_distance_point_to_line_segment(
            ball_center, line_start, line_end
        )

        # Ball intersects if distance is less than ball radius
        return distance < ball.radius
