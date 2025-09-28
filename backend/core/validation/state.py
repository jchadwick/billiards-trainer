"""Game state validation algorithms."""


class StateValidator:
    """Game state validation."""

    def __init__(self):
        pass

    def validate_positions(self, balls, table) -> tuple[bool, list[str]]:
        """Validate ball positions."""
        pass

    def validate_physics(
        self, current_state, previous_state, dt: float
    ) -> tuple[bool, list[str]]:
        """Validate physical consistency."""
        pass
