"""Configuration profile management module.

Provides advanced profile management including:
- ProfileManager: Comprehensive profile management with inheritance and conditions
- ProfileConditions: Condition evaluation for automatic profile selection
"""

from .conditions import ProfileConditions, ProfileConditionsError
from .manager import ProfileManager, ProfileManagerError

__all__ = [
    "ProfileManager",
    "ProfileManagerError",
    "ProfileConditions",
    "ProfileConditionsError",
]
