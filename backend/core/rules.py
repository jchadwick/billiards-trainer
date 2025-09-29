"""Game rules engine for different billiards games."""

from typing import Optional

from .models import BallState, GameState, GameType


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

    def _validate_nine_ball_shot(
        self, game_state: GameState, target_ball: BallState
    ) -> bool:
        """Validate a shot for 9-ball."""
        lowest_ball = self._get_lowest_numbered_ball(game_state)
        return target_ball == lowest_ball

    def _validate_eight_ball_shot(
        self, game_state: GameState, target_ball: BallState
    ) -> bool:
        """Validate a shot for 8-ball."""
        if not game_state.current_player or not target_ball or not target_ball.number:
            return False

        # Get player's assigned group
        player_group = self._get_player_group(game_state, game_state.current_player)

        # If groups not yet determined (break shot), any ball except 8 is valid
        if player_group == "undetermined":
            return target_ball.number != 8

        # If player has cleared their group, can only shoot 8-ball
        if self._player_group_cleared(game_state, game_state.current_player):
            return target_ball.number == 8

        # Otherwise, must shoot at player's assigned group
        if player_group == "solids":
            return 1 <= target_ball.number <= 7
        elif player_group == "stripes":
            return 9 <= target_ball.number <= 15

        return False

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
        return any(ball.number == 9 and ball.is_pocketed for ball in game_state.balls)

    def _is_eight_ball_over(self, game_state: GameState) -> bool:
        """Check if 8-ball game is over."""
        # Find the 8-ball
        eight_ball = None
        for ball in game_state.balls:
            if ball.number == 8:
                eight_ball = ball
                break

        if not eight_ball:
            return False

        # Game is over if 8-ball is pocketed
        return eight_ball.is_pocketed

    def _get_player_group(self, game_state: GameState, player: int) -> str:
        """Get the ball group assigned to a player ('solids', 'stripes', or 'undetermined')."""
        # Check if player has pocketed any balls yet to determine their group
        pocketed_solids = sum(
            1
            for ball in game_state.balls
            if ball.number and 1 <= ball.number <= 7 and ball.is_pocketed
        )
        pocketed_stripes = sum(
            1
            for ball in game_state.balls
            if ball.number and 9 <= ball.number <= 15 and ball.is_pocketed
        )

        # If neither player has pocketed group balls, groups are undetermined
        if pocketed_solids == 0 and pocketed_stripes == 0:
            return "undetermined"

        # Assign groups based on which balls were pocketed first
        # For simplicity, assume player 1 gets solids if any solids were pocketed
        if player == 1:
            return "solids" if pocketed_solids > 0 else "stripes"
        else:
            return "stripes" if pocketed_stripes > 0 else "solids"

    def _player_group_cleared(self, game_state: GameState, player: int) -> bool:
        """Check if player has cleared all balls in their assigned group."""
        player_group = self._get_player_group(game_state, player)

        if player_group == "undetermined":
            return False

        if player_group == "solids":
            # Check if all solid balls (1-7) are pocketed
            remaining_solids = sum(
                1
                for ball in game_state.balls
                if ball.number and 1 <= ball.number <= 7 and not ball.is_pocketed
            )
            return remaining_solids == 0
        elif player_group == "stripes":
            # Check if all stripe balls (9-15) are pocketed
            remaining_stripes = sum(
                1
                for ball in game_state.balls
                if ball.number and 9 <= ball.number <= 15 and not ball.is_pocketed
            )
            return remaining_stripes == 0

        return False

    def detect_fouls(
        self, game_state: GameState, first_ball_hit: BallState = None
    ) -> list[str]:
        """Detect fouls in the current game state for 8-ball."""
        if self.game_type != GameType.EIGHT_BALL:
            return []

        fouls = []

        # Check if cue ball was pocketed (scratch)
        cue_ball = game_state.get_cue_ball()
        if cue_ball and cue_ball.is_pocketed:
            fouls.append("cue_ball_pocketed")

        # Check if wrong ball was hit first
        if (
            game_state.current_player
            and first_ball_hit
            and not self._validate_eight_ball_shot(game_state, first_ball_hit)
        ):
            fouls.append("wrong_ball_hit_first")

        # Check if 8-ball was pocketed prematurely
        eight_ball = next((ball for ball in game_state.balls if ball.number == 8), None)
        if (
            eight_ball
            and eight_ball.is_pocketed
            and game_state.current_player
            and not self._player_group_cleared(game_state, game_state.current_player)
        ):
            fouls.append("eight_ball_pocketed_early")

        # Check if no balls were pocketed and no rail contact (basic implementation)
        # This would require more game event tracking for full implementation

        return fouls

    def get_winner(self, game_state: GameState) -> Optional[int]:
        """Determine the winner of an 8-ball game."""
        if not self._is_eight_ball_over(game_state):
            return None

        eight_ball = next((ball for ball in game_state.balls if ball.number == 8), None)
        if not eight_ball or not eight_ball.is_pocketed:
            return None

        # If 8-ball is pocketed, check conditions
        if game_state.current_player:
            # Current player wins if they cleared their group first
            if self._player_group_cleared(game_state, game_state.current_player):
                return game_state.current_player
            else:
                # Current player loses for pocketing 8-ball early
                return 2 if game_state.current_player == 1 else 1

        return None

    def should_continue_turn(
        self, game_state: GameState, balls_pocketed: list[BallState]
    ) -> bool:
        """Determine if the current player should continue their turn."""
        if self.game_type != GameType.EIGHT_BALL:
            return False

        # Player continues if they pocketed legal balls without fouls
        fouls = self.detect_fouls(game_state)
        if fouls:
            return False

        # Check if any legal balls were pocketed
        if not balls_pocketed:
            return False

        player_group = self._get_player_group(game_state, game_state.current_player)

        for ball in balls_pocketed:
            if not ball.number:
                continue

            # Legal ball pocketed - continue turn
            if (
                (player_group == "solids" and 1 <= ball.number <= 7)
                or (player_group == "stripes" and 9 <= ball.number <= 15)
                or (
                    player_group != "undetermined"
                    and ball.number == 8
                    and self._player_group_cleared(
                        game_state, game_state.current_player
                    )
                )
            ):
                return True

        return False
