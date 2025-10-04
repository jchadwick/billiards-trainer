"""System Integration Module - Main orchestrator and monitoring.

This module provides the primary interface for:
- System startup/shutdown coordination
- Module health monitoring and auto-recovery
- Performance monitoring and metrics
- System configuration management
- Production deployment support
"""

import logging

from .health_monitor import HealthMonitor
from .monitoring import SystemMetrics
from .utils import SystemUtils

# Set up logging
logger = logging.getLogger(__name__)


# Export main classes
__all__ = [
    # Health monitoring
    "HealthMonitor",
    # Performance monitoring
    "SystemMetrics",
    # Utilities
    "SystemUtils",
]
