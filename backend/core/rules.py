"""Game rules module stub.

Minimal implementation to satisfy imports. Full implementation TBD.
"""

from typing import Optional

from .models import GameType


class GameRules:
    """Game rules validator - stub implementation."""

    def __init__(self, game_type: GameType):
        """Initialize rules for given game type.

        Args:
            game_type: Type of billiards game
        """
        self.game_type = game_type

    def validate_shot(self, game_state, target_ball) -> bool:
        """Validate if a shot is legal.

        Args:
            game_state: Current game state
            target_ball: Target ball for the shot

        Returns:
            True if shot is legal (stub always returns True)
        """
        return True

    def check_game_over(self, game_state) -> bool:
        """Check if game is over.

        Args:
            game_state: Current game state

        Returns:
            True if game is over (stub always returns False)
        """
        return False
