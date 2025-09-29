"""Game rules engine for different billiards games."""

from .models import GameState, GameType, BallState

class GameRules:
    """A class to manage game rules for different billiards games."""

    def __init__(self, game_type: GameType):
        self.game_type = game_type

    def validate_shot(self, game_state: GameState, target_ball: BallState) -> bool:
        """Validate a shot based on the game rules."""
        if self.game_type == GameType.NINE_BALL:
            return self._validate_nine_ball_shot(game_state, target_ball)
        elif self.game_type == GameType.EIGHT_BALL:
            return self._validate_eight_ball_shot(game_state, target_ball)
        else:
            return True

    def _validate_nine_ball_shot(self, game_state: GameState, target_ball: BallState) -> bool:
        """Validate a shot for 9-ball."""
        lowest_ball = self._get_lowest_numbered_ball(game_state)
        return target_ball == lowest_ball

    def _validate_eight_ball_shot(self, game_state: GameState, target_ball: BallState) -> bool:
        """Validate a shot for 8-ball."""
        # Implementation for 8-ball rules
        return True

    def _get_lowest_numbered_ball(self, game_state: GameState) -> BallState:
        """Get the lowest numbered ball on the table."""
        lowest_ball = None
        for ball in game_state.balls:
            if not ball.is_cue_ball and not ball.is_pocketed:
                if lowest_ball is None or ball.number < lowest_ball.number:
                    lowest_ball = ball
        return lowest_ball

    def check_game_over(self, game_state: GameState) -> bool:
        """Check if the game is over."""
        if self.game_type == GameType.NINE_BALL:
            return self._is_nine_ball_over(game_state)
        elif self.game_type == GameType.EIGHT_BALL:
            return self._is_eight_ball_over(game_state)
        return False

    def _is_nine_ball_over(self, game_state: GameState) -> bool:
        """Check if 9-ball game is over."""
        for ball in game_state.balls:
            if ball.number == 9 and ball.is_pocketed:
                return True
        return False

    def _is_eight_ball_over(self, game_state: GameState) -> bool:
        """Check if 8-ball game is over."""
        # Implementation for 8-ball game over condition
        return False
