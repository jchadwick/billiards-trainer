"""Common models and utilities shared across backend modules.

This package provides shared data models that are used across multiple backend
modules (vision, core, tracking, etc.) to ensure consistency and prevent circular
dependencies.

Key exports:
- Ball: Unified ball model with coordinate metadata
"""

from .models import Ball

__all__ = ["Ball"]
