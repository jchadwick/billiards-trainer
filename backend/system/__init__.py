"""System Integration Module - Main orchestrator and monitoring.

This module provides the primary interface for:
- System startup/shutdown coordination
- Module health monitoring and auto-recovery
- Performance monitoring and metrics
- System configuration management
- Production deployment support
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from .health import HealthMonitor, HealthStatus, ModuleHealth, SystemHealth

if TYPE_CHECKING:
    from .orchestrator import SystemOrchestrator
from .monitoring import (
    AlertManager,
    MetricsCollector,
    PerformanceDashboard,
    PerformanceMonitor,
    SystemMetrics,
)
from .recovery import RecoveryAction, RecoveryManager, RecoveryPolicy
from .utils import ProcessManager, ResourceMonitor, SystemUtils

# Set up logging
logger = logging.getLogger(__name__)


# Lazy import for orchestrator to avoid circular imports
def create_orchestrator(
    config: Optional[dict[str, Any]] = None
) -> "SystemOrchestrator":
    """Create system orchestrator with lazy import."""
    from .orchestrator import SystemConfig, SystemOrchestrator

    if config is None:
        config_obj = SystemConfig()
    else:
        config_obj = SystemConfig(**config)

    return SystemOrchestrator(config_obj)


# Export main classes
__all__ = [
    # Main orchestrator (lazy import function)
    "create_orchestrator",
    # Health monitoring
    "HealthMonitor",
    "HealthStatus",
    "ModuleHealth",
    "SystemHealth",
    # Performance monitoring
    "PerformanceMonitor",
    "PerformanceDashboard",
    "MetricsCollector",
    "AlertManager",
    "SystemMetrics",
    # Recovery management
    "RecoveryManager",
    "RecoveryPolicy",
    "RecoveryAction",
    # Utilities
    "SystemUtils",
    "ResourceMonitor",
    "ProcessManager",
]
