"""Health monitoring system for tracking module and system health."""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""

    name: str
    status: HealthStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    response_time: float = 0.0


@dataclass
class ModuleHealth:
    """Health status of a system module."""

    module_name: str
    status: HealthStatus
    checks: list[HealthCheck] = field(default_factory=list)
    last_check: float = field(default_factory=time.time)
    uptime: float = 0.0
    restart_count: int = 0
    error_count: int = 0


@dataclass
class SystemHealth:
    """Overall system health status."""

    overall_status: HealthStatus
    modules: dict[str, ModuleHealth] = field(default_factory=dict)
    system_uptime: float = 0.0
    last_check: float = field(default_factory=time.time)
    total_errors: int = 0
    performance_score: float = 1.0


class HealthMonitor:
    """System health monitoring and reporting."""

    def __init__(self):
        """Initialize health monitor."""
        self.modules: dict[str, Any] = {}
        self.module_health: dict[str, ModuleHealth] = {}
        self.system_start_time = time.time()
        self.is_running = False
        self.check_interval = 30.0

        # Health check functions for each module type
        self.health_checkers = {
            "config": self._check_config_health,
            "core": self._check_core_health,
            "vision": self._check_vision_health,
            "projector": self._check_projector_health,
            "api": self._check_api_health,
        }

        logger.info("Health Monitor initialized")

    async def start(self) -> None:
        """Start health monitoring."""
        self.is_running = True
        logger.info("Health monitoring started")

    async def stop(self) -> None:
        """Stop health monitoring."""
        self.is_running = False
        logger.info("Health monitoring stopped")

    async def register_module(self, module_name: str, module_instance: Any) -> None:
        """Register a module for health monitoring.

        Args:
            module_name: Name of the module
            module_instance: Instance of the module
        """
        self.modules[module_name] = module_instance
        self.module_health[module_name] = ModuleHealth(
            module_name=module_name, status=HealthStatus.UNKNOWN
        )

        logger.info(f"Module {module_name} registered for health monitoring")

    async def unregister_module(self, module_name: str) -> None:
        """Unregister a module from health monitoring.

        Args:
            module_name: Name of the module to unregister
        """
        if module_name in self.modules:
            del self.modules[module_name]

        if module_name in self.module_health:
            del self.module_health[module_name]

        logger.info(f"Module {module_name} unregistered from health monitoring")

    async def check_module_health(self, module_name: str) -> ModuleHealth:
        """Check health of a specific module.

        Args:
            module_name: Name of module to check

        Returns:
            Module health status
        """
        if module_name not in self.modules:
            return ModuleHealth(
                module_name=module_name,
                status=HealthStatus.UNKNOWN,
                checks=[
                    HealthCheck(
                        name="existence",
                        status=HealthStatus.UNHEALTHY,
                        message="Module not registered",
                    )
                ],
            )

        try:
            start_time = time.time()
            module = self.modules[module_name]

            # Get health checker function
            checker = self.health_checkers.get(module_name, self._check_generic_health)

            # Perform health checks
            checks = await checker(module)

            # Calculate overall module status
            overall_status = self._calculate_module_status(checks)

            # Update module health
            health = ModuleHealth(
                module_name=module_name,
                status=overall_status,
                checks=checks,
                last_check=time.time(),
                uptime=time.time() - self.system_start_time,
            )

            # Update stored health
            if module_name in self.module_health:
                old_health = self.module_health[module_name]
                health.restart_count = old_health.restart_count
                health.error_count = old_health.error_count

                # Increment error count if unhealthy
                if overall_status == HealthStatus.UNHEALTHY:
                    health.error_count += 1

            self.module_health[module_name] = health

            response_time = time.time() - start_time
            logger.debug(
                f"Health check for {module_name} completed in {response_time:.3f}s"
            )

            return health

        except Exception as e:
            logger.error(f"Health check failed for {module_name}: {e}")
            error_health = ModuleHealth(
                module_name=module_name,
                status=HealthStatus.UNHEALTHY,
                checks=[
                    HealthCheck(
                        name="health_check",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed: {e}",
                    )
                ],
                last_check=time.time(),
            )

            self.module_health[module_name] = error_health
            return error_health

    async def get_system_health(self) -> SystemHealth:
        """Get overall system health status.

        Returns:
            Complete system health status
        """
        try:
            # Check health of all registered modules
            for module_name in self.modules:
                await self.check_module_health(module_name)

            # Calculate overall system status
            overall_status = self._calculate_system_status()

            # Calculate performance score
            performance_score = self._calculate_performance_score()

            # Count total errors
            total_errors = sum(
                health.error_count for health in self.module_health.values()
            )

            return SystemHealth(
                overall_status=overall_status,
                modules=self.module_health.copy(),
                system_uptime=time.time() - self.system_start_time,
                last_check=time.time(),
                total_errors=total_errors,
                performance_score=performance_score,
            )

        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return SystemHealth(
                overall_status=HealthStatus.UNHEALTHY,
                system_uptime=time.time() - self.system_start_time,
                last_check=time.time(),
            )

    # Module-specific health checkers

    async def _check_config_health(self, module) -> list[HealthCheck]:
        """Check configuration module health."""
        checks = []

        try:
            # Check if module is properly initialized
            if hasattr(module, "get_configuration"):
                start_time = time.time()
                config = await module.get_configuration()
                response_time = time.time() - start_time

                if config:
                    checks.append(
                        HealthCheck(
                            name="configuration_access",
                            status=HealthStatus.HEALTHY,
                            message="Configuration accessible",
                            response_time=response_time,
                        )
                    )
                else:
                    checks.append(
                        HealthCheck(
                            name="configuration_access",
                            status=HealthStatus.UNHEALTHY,
                            message="Configuration not available",
                        )
                    )
            else:
                checks.append(
                    HealthCheck(
                        name="interface_check",
                        status=HealthStatus.UNHEALTHY,
                        message="Configuration interface not available",
                    )
                )

        except Exception as e:
            checks.append(
                HealthCheck(
                    name="configuration_access",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Configuration check failed: {e}",
                )
            )

        return checks

    async def _check_core_health(self, module) -> list[HealthCheck]:
        """Check core module health."""
        checks = []

        try:
            # Check if module has current state
            if hasattr(module, "get_current_state"):
                state = module.get_current_state()
                checks.append(
                    HealthCheck(
                        name="state_access",
                        status=HealthStatus.HEALTHY if state else HealthStatus.DEGRADED,
                        message="State accessible" if state else "No current state",
                    )
                )

            # Check performance metrics
            if hasattr(module, "get_performance_metrics"):
                metrics = module.get_performance_metrics()
                if metrics.errors_count > 100:  # High error threshold
                    checks.append(
                        HealthCheck(
                            name="error_rate",
                            status=HealthStatus.DEGRADED,
                            message=f"High error count: {metrics.errors_count}",
                            details={"error_count": metrics.errors_count},
                        )
                    )
                else:
                    checks.append(
                        HealthCheck(
                            name="error_rate",
                            status=HealthStatus.HEALTHY,
                            message="Error rate acceptable",
                            details={"error_count": metrics.errors_count},
                        )
                    )

            # Check processing performance
            if hasattr(module, "get_performance_metrics"):
                metrics = module.get_performance_metrics()
                if metrics.avg_update_time > 1.0:  # Slow processing
                    checks.append(
                        HealthCheck(
                            name="processing_speed",
                            status=HealthStatus.DEGRADED,
                            message=f"Slow processing: {metrics.avg_update_time:.3f}s",
                            details={"avg_update_time": metrics.avg_update_time},
                        )
                    )
                else:
                    checks.append(
                        HealthCheck(
                            name="processing_speed",
                            status=HealthStatus.HEALTHY,
                            message="Processing speed good",
                            details={"avg_update_time": metrics.avg_update_time},
                        )
                    )

        except Exception as e:
            checks.append(
                HealthCheck(
                    name="core_health",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Core health check failed: {e}",
                )
            )

        return checks

    async def _check_vision_health(self, module) -> list[HealthCheck]:
        """Check vision module health."""
        checks = []

        try:
            # Check if capture is running
            if hasattr(module, "_is_running"):
                if module._is_running:
                    checks.append(
                        HealthCheck(
                            name="capture_status",
                            status=HealthStatus.HEALTHY,
                            message="Camera capture running",
                        )
                    )
                else:
                    checks.append(
                        HealthCheck(
                            name="capture_status",
                            status=HealthStatus.DEGRADED,
                            message="Camera capture not running",
                        )
                    )

            # Check camera connection
            if hasattr(module, "camera") and hasattr(module.camera, "is_connected"):
                if module.camera.is_connected():
                    checks.append(
                        HealthCheck(
                            name="camera_connection",
                            status=HealthStatus.HEALTHY,
                            message="Camera connected",
                        )
                    )
                else:
                    checks.append(
                        HealthCheck(
                            name="camera_connection",
                            status=HealthStatus.UNHEALTHY,
                            message="Camera not connected",
                        )
                    )

            # Check processing statistics
            if hasattr(module, "get_statistics"):
                stats = module.get_statistics()

                # Check FPS
                if stats.get("avg_fps", 0) > 15:
                    checks.append(
                        HealthCheck(
                            name="frame_rate",
                            status=HealthStatus.HEALTHY,
                            message=f"Good frame rate: {stats['avg_fps']:.1f} FPS",
                            details={"fps": stats["avg_fps"]},
                        )
                    )
                elif stats.get("avg_fps", 0) > 5:
                    checks.append(
                        HealthCheck(
                            name="frame_rate",
                            status=HealthStatus.DEGRADED,
                            message=f"Low frame rate: {stats['avg_fps']:.1f} FPS",
                            details={"fps": stats["avg_fps"]},
                        )
                    )
                else:
                    checks.append(
                        HealthCheck(
                            name="frame_rate",
                            status=HealthStatus.UNHEALTHY,
                            message=f"Very low frame rate: {stats['avg_fps']:.1f} FPS",
                            details={"fps": stats["avg_fps"]},
                        )
                    )

                # Check dropped frames
                dropped_rate = stats.get("frames_dropped", 0) / max(
                    stats.get("frames_processed", 1), 1
                )
                if dropped_rate < 0.1:
                    checks.append(
                        HealthCheck(
                            name="frame_drops",
                            status=HealthStatus.HEALTHY,
                            message=f"Low drop rate: {dropped_rate:.1%}",
                            details={"drop_rate": dropped_rate},
                        )
                    )
                else:
                    checks.append(
                        HealthCheck(
                            name="frame_drops",
                            status=HealthStatus.DEGRADED,
                            message=f"High drop rate: {dropped_rate:.1%}",
                            details={"drop_rate": dropped_rate},
                        )
                    )

        except Exception as e:
            checks.append(
                HealthCheck(
                    name="vision_health",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Vision health check failed: {e}",
                )
            )

        return checks

    async def _check_projector_health(self, module) -> list[HealthCheck]:
        """Check projector module health."""
        checks = []

        try:
            # Check if projector is initialized
            if hasattr(module, "is_initialized") and module.is_initialized:
                checks.append(
                    HealthCheck(
                        name="initialization",
                        status=HealthStatus.HEALTHY,
                        message="Projector initialized",
                    )
                )
            else:
                checks.append(
                    HealthCheck(
                        name="initialization",
                        status=HealthStatus.DEGRADED,
                        message="Projector not initialized",
                    )
                )

            # Check network connectivity if applicable
            if hasattr(module, "network_manager"):
                # This would depend on the actual projector implementation
                checks.append(
                    HealthCheck(
                        name="network_connection",
                        status=HealthStatus.HEALTHY,
                        message="Network connection available",
                    )
                )

        except Exception as e:
            checks.append(
                HealthCheck(
                    name="projector_health",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Projector health check failed: {e}",
                )
            )

        return checks

    async def _check_api_health(self, module) -> list[HealthCheck]:
        """Check API module health."""
        checks = []

        try:
            # Basic API app validation
            if hasattr(module, "router") or hasattr(module, "routes"):
                checks.append(
                    HealthCheck(
                        name="app_structure",
                        status=HealthStatus.HEALTHY,
                        message="API app structure valid",
                    )
                )
            else:
                checks.append(
                    HealthCheck(
                        name="app_structure",
                        status=HealthStatus.DEGRADED,
                        message="API app structure unclear",
                    )
                )

        except Exception as e:
            checks.append(
                HealthCheck(
                    name="api_health",
                    status=HealthStatus.UNHEALTHY,
                    message=f"API health check failed: {e}",
                )
            )

        return checks

    async def _check_generic_health(self, module) -> list[HealthCheck]:
        """Generic health check for unknown module types."""
        checks = []

        try:
            # Basic existence check
            checks.append(
                HealthCheck(
                    name="module_exists",
                    status=HealthStatus.HEALTHY,
                    message="Module instance exists",
                )
            )

            # Check if module has common health methods
            if hasattr(module, "get_status"):
                status = module.get_status()
                checks.append(
                    HealthCheck(
                        name="status_method",
                        status=HealthStatus.HEALTHY,
                        message="Status method available",
                        details={"status": str(status)},
                    )
                )

        except Exception as e:
            checks.append(
                HealthCheck(
                    name="generic_health",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Generic health check failed: {e}",
                )
            )

        return checks

    def _calculate_module_status(self, checks: list[HealthCheck]) -> HealthStatus:
        """Calculate overall module status from individual checks.

        Args:
            checks: List of health checks

        Returns:
            Overall module health status
        """
        if not checks:
            return HealthStatus.UNKNOWN

        unhealthy_count = sum(
            1 for check in checks if check.status == HealthStatus.UNHEALTHY
        )
        degraded_count = sum(
            1 for check in checks if check.status == HealthStatus.DEGRADED
        )

        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _calculate_system_status(self) -> HealthStatus:
        """Calculate overall system status from module statuses."""
        if not self.module_health:
            return HealthStatus.UNKNOWN

        statuses = [health.status for health in self.module_health.values()]

        unhealthy_count = sum(
            1 for status in statuses if status == HealthStatus.UNHEALTHY
        )
        degraded_count = sum(
            1 for status in statuses if status == HealthStatus.DEGRADED
        )

        # System is unhealthy if any critical module is unhealthy
        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        # System is degraded if any module is degraded
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _calculate_performance_score(self) -> float:
        """Calculate system performance score (0.0 to 1.0)."""
        if not self.module_health:
            return 0.0

        total_score = 0.0
        module_count = len(self.module_health)

        for health in self.module_health.values():
            if health.status == HealthStatus.HEALTHY:
                total_score += 1.0
            elif health.status == HealthStatus.DEGRADED:
                total_score += 0.5
            # UNHEALTHY contributes 0.0

        return total_score / module_count if module_count > 0 else 0.0
