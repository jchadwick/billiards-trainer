"""Auto-recovery and fault tolerance system."""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from .health import HealthStatus

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """Types of recovery actions."""

    RESTART_MODULE = "restart_module"
    RESTART_SYSTEM = "restart_system"
    RESET_STATE = "reset_state"
    CLEAR_CACHE = "clear_cache"
    FAILOVER = "failover"
    ALERT_ONLY = "alert_only"


@dataclass
class RecoveryPolicy:
    """Recovery policy configuration."""

    name: str
    condition: str  # Description of when to trigger
    action: RecoveryAction
    max_attempts: int = 3
    backoff_seconds: float = 30.0
    requires_confirmation: bool = False
    critical: bool = False


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""

    policy: RecoveryPolicy
    target: str  # Module or system component
    action: RecoveryAction
    timestamp: float = field(default_factory=time.time)
    success: bool = False
    error_message: Optional[str] = None
    duration: float = 0.0


class RecoveryManager:
    """Manages automatic recovery and fault tolerance."""

    def __init__(self):
        """Initialize recovery manager."""
        self.policies: dict[str, RecoveryPolicy] = {}
        self.recovery_history: list[RecoveryAttempt] = []
        self.recovery_callbacks: dict[str, Callable] = {}
        self.is_enabled = True

        # Default recovery policies
        self._setup_default_policies()

        logger.info("Recovery Manager initialized")

    def _setup_default_policies(self) -> None:
        """Setup default recovery policies."""
        # Vision module recovery
        self.policies["vision_unhealthy"] = RecoveryPolicy(
            name="Vision Module Recovery",
            condition="Vision module becomes unhealthy",
            action=RecoveryAction.RESTART_MODULE,
            max_attempts=3,
            backoff_seconds=30.0,
        )

        # Core module recovery
        self.policies["core_unhealthy"] = RecoveryPolicy(
            name="Core Module Recovery",
            condition="Core module becomes unhealthy",
            action=RecoveryAction.RESET_STATE,
            max_attempts=2,
            backoff_seconds=60.0,
        )

        # Memory pressure recovery
        self.policies["high_memory"] = RecoveryPolicy(
            name="High Memory Usage Recovery",
            condition="Memory usage exceeds 90%",
            action=RecoveryAction.CLEAR_CACHE,
            max_attempts=1,
            backoff_seconds=10.0,
        )

        # System-wide recovery
        self.policies["system_critical"] = RecoveryPolicy(
            name="System Critical Recovery",
            condition="Multiple modules unhealthy",
            action=RecoveryAction.RESTART_SYSTEM,
            max_attempts=1,
            backoff_seconds=300.0,
            requires_confirmation=True,
            critical=True,
        )

        logger.info("Default recovery policies configured")

    def register_callback(self, action: RecoveryAction, callback: Callable) -> None:
        """Register a callback for recovery actions.

        Args:
            action: Recovery action type
            callback: Function to call for this action
        """
        self.recovery_callbacks[action.value] = callback
        logger.debug(f"Recovery callback registered for {action.value}")

    async def evaluate_recovery(
        self,
        module_name: str,
        health_status: HealthStatus,
        context: dict[str, Any] = None,
    ) -> bool:
        """Evaluate if recovery is needed and execute if appropriate.

        Args:
            module_name: Name of the module
            health_status: Current health status
            context: Additional context for recovery decisions

        Returns:
            True if recovery was attempted
        """
        if not self.is_enabled:
            return False

        context = context or {}

        try:
            # Determine appropriate policy
            policy = self._select_policy(module_name, health_status, context)

            if policy is None:
                return False

            # Check if we should attempt recovery
            if not self._should_attempt_recovery(module_name, policy):
                logger.warning(
                    f"Recovery for {module_name} skipped - too many recent attempts"
                )
                return False

            # Execute recovery
            logger.info(
                f"Attempting recovery for {module_name} using policy: {policy.name}"
            )
            success = await self._execute_recovery(module_name, policy, context)

            # Record attempt
            attempt = RecoveryAttempt(
                policy=policy, target=module_name, action=policy.action, success=success
            )
            self.recovery_history.append(attempt)

            # Limit history size
            if len(self.recovery_history) > 1000:
                self.recovery_history = self.recovery_history[-500:]

            return success

        except Exception as e:
            logger.error(f"Recovery evaluation failed for {module_name}: {e}")
            return False

    async def force_recovery(
        self, module_name: str, action: RecoveryAction, context: dict[str, Any] = None
    ) -> bool:
        """Force a specific recovery action.

        Args:
            module_name: Target module name
            action: Recovery action to perform
            context: Additional context

        Returns:
            True if recovery was successful
        """
        try:
            logger.info(f"Forcing recovery action {action.value} for {module_name}")

            # Create temporary policy
            policy = RecoveryPolicy(
                name=f"Forced {action.value}",
                condition="Manual intervention",
                action=action,
                max_attempts=1,
            )

            success = await self._execute_recovery(module_name, policy, context or {})

            # Record attempt
            attempt = RecoveryAttempt(
                policy=policy, target=module_name, action=action, success=success
            )
            self.recovery_history.append(attempt)

            return success

        except Exception as e:
            logger.error(f"Forced recovery failed for {module_name}: {e}")
            return False

    def get_recovery_history(
        self, module_name: Optional[str] = None, limit: Optional[int] = None
    ) -> list[RecoveryAttempt]:
        """Get recovery attempt history.

        Args:
            module_name: Filter by module name (all if None)
            limit: Maximum number of entries to return

        Returns:
            List of recovery attempts
        """
        history = self.recovery_history

        if module_name:
            history = [attempt for attempt in history if attempt.target == module_name]

        if limit:
            history = history[-limit:]

        return history

    def get_recovery_stats(self) -> dict[str, Any]:
        """Get recovery statistics.

        Returns:
            Recovery statistics summary
        """
        total_attempts = len(self.recovery_history)
        successful_attempts = sum(
            1 for attempt in self.recovery_history if attempt.success
        )

        # Group by action type
        action_stats = {}
        for attempt in self.recovery_history:
            action = attempt.action.value
            if action not in action_stats:
                action_stats[action] = {"total": 0, "successful": 0}

            action_stats[action]["total"] += 1
            if attempt.success:
                action_stats[action]["successful"] += 1

        # Recent failures (last hour)
        recent_time = time.time() - 3600
        recent_failures = sum(
            1
            for attempt in self.recovery_history
            if attempt.timestamp > recent_time and not attempt.success
        )

        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": (
                successful_attempts / total_attempts if total_attempts > 0 else 0.0
            ),
            "action_stats": action_stats,
            "recent_failures": recent_failures,
            "is_enabled": self.is_enabled,
        }

    def enable_recovery(self) -> None:
        """Enable automatic recovery."""
        self.is_enabled = True
        logger.info("Auto-recovery enabled")

    def disable_recovery(self) -> None:
        """Disable automatic recovery."""
        self.is_enabled = False
        logger.warning("Auto-recovery disabled")

    def _select_policy(
        self, module_name: str, health_status: HealthStatus, context: dict[str, Any]
    ) -> Optional[RecoveryPolicy]:
        """Select appropriate recovery policy.

        Args:
            module_name: Module name
            health_status: Current health status
            context: Additional context

        Returns:
            Selected recovery policy or None
        """
        # Only attempt recovery for unhealthy modules
        if health_status != HealthStatus.UNHEALTHY:
            return None

        # Module-specific policies
        policy_key = f"{module_name}_unhealthy"
        if policy_key in self.policies:
            return self.policies[policy_key]

        # Check for system-wide issues
        if context.get("multiple_modules_unhealthy"):
            return self.policies.get("system_critical")

        # Check resource constraints
        if context.get("memory_critical"):
            return self.policies.get("high_memory")

        # Default: restart module
        return RecoveryPolicy(
            name=f"Default {module_name} Recovery",
            condition="Module unhealthy",
            action=RecoveryAction.RESTART_MODULE,
            max_attempts=2,
        )

    def _should_attempt_recovery(
        self, module_name: str, policy: RecoveryPolicy
    ) -> bool:
        """Check if recovery should be attempted based on history.

        Args:
            module_name: Module name
            policy: Recovery policy

        Returns:
            True if recovery should be attempted
        """
        # Get recent attempts for this module/policy
        recent_time = time.time() - policy.backoff_seconds
        recent_attempts = [
            attempt
            for attempt in self.recovery_history
            if (
                attempt.target == module_name
                and attempt.policy.name == policy.name
                and attempt.timestamp > recent_time
            )
        ]

        # Check if we've exceeded max attempts
        return not len(recent_attempts) >= policy.max_attempts

    async def _execute_recovery(
        self, module_name: str, policy: RecoveryPolicy, context: dict[str, Any]
    ) -> bool:
        """Execute a recovery action.

        Args:
            module_name: Target module
            policy: Recovery policy
            context: Additional context

        Returns:
            True if recovery was successful
        """
        start_time = time.time()

        try:
            action = policy.action

            # Get callback for this action
            callback = self.recovery_callbacks.get(action.value)

            if callback is None:
                logger.error(
                    f"No callback registered for recovery action: {action.value}"
                )
                return False

            # Execute callback
            if action == RecoveryAction.RESTART_MODULE:
                success = await callback(module_name)
            elif action == RecoveryAction.RESTART_SYSTEM:
                success = await callback()
            elif (
                action == RecoveryAction.RESET_STATE
                or action == RecoveryAction.CLEAR_CACHE
            ):
                success = await callback(module_name)
            elif action == RecoveryAction.FAILOVER:
                success = await callback(module_name, context)
            else:
                # ALERT_ONLY or unknown action
                logger.warning(f"Recovery action {action.value} executed (alert only)")
                success = True

            duration = time.time() - start_time

            if success:
                logger.info(f"Recovery successful for {module_name} in {duration:.2f}s")
            else:
                logger.error(f"Recovery failed for {module_name} after {duration:.2f}s")

            return success

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Recovery execution failed for {module_name} after {duration:.2f}s: {e}"
            )
            return False

    async def check_system_recovery_needed(self, system_health) -> bool:
        """Check if system-wide recovery is needed.

        Args:
            system_health: Current system health status

        Returns:
            True if system recovery is needed
        """
        try:
            unhealthy_modules = [
                name
                for name, health in system_health.modules.items()
                if health.status == HealthStatus.UNHEALTHY
            ]

            # If multiple critical modules are unhealthy, consider system recovery
            critical_modules = {"core", "vision", "api"}
            unhealthy_critical = [
                mod for mod in unhealthy_modules if mod in critical_modules
            ]

            if len(unhealthy_critical) >= 2:
                logger.warning(
                    f"Multiple critical modules unhealthy: {unhealthy_critical}"
                )
                return True

            # Check overall system performance
            if system_health.performance_score < 0.3:
                logger.warning(
                    f"System performance critically low: {system_health.performance_score}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking system recovery need: {e}")
            return False
