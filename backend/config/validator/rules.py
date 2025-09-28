"""Validation rules for configuration."""


class ValidationRules:
    """Configuration validation rules."""

    def __init__(self):
        pass

    def check_range(self, value, min_val=None, max_val=None) -> bool:
        """Check if value is within range."""
        pass

    def check_type(self, value, expected_type) -> bool:
        """Check value type."""
        pass
