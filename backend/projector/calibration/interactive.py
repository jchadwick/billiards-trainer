"""Interactive calibration interface."""


class InteractiveCalibration:
    """Interactive projector calibration."""

    def __init__(self):
        """Initialize interactive calibration."""
        pass

    def start_calibration(self) -> None:
        """Start interactive calibration procedure."""
        pass

    def display_calibration_grid(self) -> None:
        """Display calibration grid pattern."""
        pass

    def adjust_corner_point(
        self, corner_index: int, new_position: tuple[float, float]
    ) -> None:
        """Adjust calibration corner point."""
        pass

    def save_calibration(self, points: list[tuple[float, float]]) -> bool:
        """Save calibration points."""
        pass
