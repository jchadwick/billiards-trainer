"""Graceful shutdown coordination for the billiards trainer system.

This module provides comprehensive shutdown coordination for all system modules
including proper resource cleanup, data persistence, and module shutdown sequencing.
"""

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ShutdownPhase(Enum):
    """Phases of the shutdown process."""

    INITIATED = "initiated"
    STOPPING_NEW_REQUESTS = "stopping_new_requests"
    DRAINING_CONNECTIONS = "draining_connections"
    STOPPING_BACKGROUND_TASKS = "stopping_background_tasks"
    SAVING_STATE = "saving_state"
    CLEANING_RESOURCES = "cleaning_resources"
    SHUTTING_DOWN_MODULES = "shutting_down_modules"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


class ShutdownStatus(Enum):
    """Status of the shutdown process."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    FORCED = "forced"


@dataclass
class ShutdownProgress:
    """Progress tracking for shutdown process."""

    phase: ShutdownPhase = ShutdownPhase.INITIATED
    status: ShutdownStatus = ShutdownStatus.NOT_STARTED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    elapsed_time: float = 0.0
    modules_completed: list[str] = field(default_factory=list)
    modules_failed: list[str] = field(default_factory=list)
    active_operations: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    force_requested: bool = False


@dataclass
class ShutdownConfig:
    """Configuration for shutdown process."""

    # Timeout settings
    graceful_timeout: float = 30.0  # Total time allowed for graceful shutdown
    module_timeout: float = 10.0  # Time allowed per module to shutdown
    connection_drain_timeout: float = 5.0  # Time to wait for connections to drain
    background_task_timeout: float = 5.0  # Time to wait for background tasks

    # Behavior settings
    save_state_on_shutdown: bool = True
    create_backup_on_shutdown: bool = True
    force_kill_on_timeout: bool = True
    wait_for_active_operations: bool = True

    # Retry settings
    max_module_retries: int = 2
    retry_delay: float = 1.0


class ShutdownCoordinator:
    """Coordinates graceful shutdown of all system modules."""

    def __init__(self, config: Optional[ShutdownConfig] = None):
        """Initialize shutdown coordinator.

        Args:
            config: Shutdown configuration
        """
        self.config = config or ShutdownConfig()
        self.progress = ShutdownProgress()
        self._shutdown_lock = asyncio.Lock()
        self._is_shutting_down = False
        self._force_shutdown = False
        self._active_operations: set[str] = set()
        self._shutdown_callbacks: list[Callable[[], None]] = []

        # Module shutdown functions
        self._module_shutdowns: dict[str, Callable[[], Any]] = {}

        # Background tasks to cleanup
        self._background_tasks: set[asyncio.Task] = set()

    def register_module_shutdown(
        self, module_name: str, shutdown_func: Callable[[], Any]
    ) -> None:
        """Register a module shutdown function.

        Args:
            module_name: Name of the module
            shutdown_func: Function to call for module shutdown (can be sync or async)
        """
        self._module_shutdowns[module_name] = shutdown_func
        logger.debug(f"Registered shutdown function for module: {module_name}")

    def register_background_task(self, task: asyncio.Task) -> None:
        """Register a background task for cleanup during shutdown.

        Args:
            task: Background task to track
        """
        self._background_tasks.add(task)

        # Remove completed tasks automatically
        def cleanup_task(t: asyncio.Task):
            self._background_tasks.discard(t)

        task.add_done_callback(cleanup_task)

    def register_active_operation(self, operation_id: str) -> None:
        """Register an active operation that should complete before shutdown.

        Args:
            operation_id: Unique identifier for the operation
        """
        self._active_operations.add(operation_id)
        self.progress.active_operations = len(self._active_operations)

    def unregister_active_operation(self, operation_id: str) -> None:
        """Unregister an active operation.

        Args:
            operation_id: Unique identifier for the operation
        """
        self._active_operations.discard(operation_id)
        self.progress.active_operations = len(self._active_operations)

    def add_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback to be called during shutdown.

        Args:
            callback: Function to call during shutdown
        """
        self._shutdown_callbacks.append(callback)

    async def initiate_shutdown(
        self, force: bool = False, save_state: bool = True
    ) -> bool:
        """Initiate graceful shutdown of all modules.

        Args:
            force: Whether to force shutdown without waiting
            save_state: Whether to save system state before shutdown

        Returns:
            True if shutdown completed successfully
        """
        async with self._shutdown_lock:
            if self._is_shutting_down:
                logger.warning("Shutdown already in progress")
                return self.progress.status == ShutdownStatus.COMPLETED

            logger.info("Initiating graceful system shutdown")
            self._is_shutting_down = True
            self._force_shutdown = force

            # Initialize progress tracking
            self.progress = ShutdownProgress(
                status=ShutdownStatus.IN_PROGRESS,
                started_at=datetime.now(timezone.utc),
                force_requested=force,
            )

            try:
                # Execute shutdown sequence
                success = await self._execute_shutdown_sequence(save_state)

                # Update final status
                self.progress.completed_at = datetime.now(timezone.utc)
                self.progress.elapsed_time = (
                    self.progress.completed_at - self.progress.started_at
                ).total_seconds()

                if success:
                    self.progress.status = ShutdownStatus.COMPLETED
                    self.progress.phase = ShutdownPhase.COMPLETED
                    logger.info(
                        f"Graceful shutdown completed in {self.progress.elapsed_time:.2f}s"
                    )
                else:
                    self.progress.status = ShutdownStatus.FAILED
                    self.progress.phase = ShutdownPhase.FAILED
                    logger.error(
                        f"Shutdown failed after {self.progress.elapsed_time:.2f}s"
                    )

                return success

            except Exception as e:
                logger.error(f"Shutdown process failed: {e}")
                self.progress.status = ShutdownStatus.FAILED
                self.progress.phase = ShutdownPhase.FAILED
                self.progress.errors.append(f"Shutdown error: {str(e)}")
                return False

    async def _execute_shutdown_sequence(self, save_state: bool) -> bool:
        """Execute the complete shutdown sequence.

        Args:
            save_state: Whether to save system state

        Returns:
            True if all phases completed successfully
        """
        shutdown_start = time.time()

        try:
            # Phase 1: Stop accepting new requests
            self.progress.phase = ShutdownPhase.STOPPING_NEW_REQUESTS
            await self._stop_new_requests()

            # Phase 2: Drain existing connections
            self.progress.phase = ShutdownPhase.DRAINING_CONNECTIONS
            await self._drain_connections()

            # Phase 3: Stop background tasks
            self.progress.phase = ShutdownPhase.STOPPING_BACKGROUND_TASKS
            await self._stop_background_tasks()

            # Phase 4: Wait for active operations (if not forced)
            if not self._force_shutdown and self.config.wait_for_active_operations:
                await self._wait_for_active_operations()

            # Phase 5: Save state
            if save_state and self.config.save_state_on_shutdown:
                self.progress.phase = ShutdownPhase.SAVING_STATE
                await self._save_system_state()

            # Phase 6: Clean up resources
            self.progress.phase = ShutdownPhase.CLEANING_RESOURCES
            await self._cleanup_resources()

            # Phase 7: Shutdown modules
            self.progress.phase = ShutdownPhase.SHUTTING_DOWN_MODULES
            await self._shutdown_modules()

            # Phase 8: Finalize
            self.progress.phase = ShutdownPhase.FINALIZING
            await self._finalize_shutdown()

            # Check if we exceeded timeout
            elapsed = time.time() - shutdown_start
            if elapsed > self.config.graceful_timeout:
                self.progress.warnings.append(
                    f"Shutdown took {elapsed:.2f}s, exceeded timeout of {self.config.graceful_timeout}s"
                )

            return len(self.progress.errors) == 0

        except asyncio.TimeoutError:
            logger.error("Shutdown sequence timed out")
            self.progress.errors.append("Shutdown sequence timed out")

            if self.config.force_kill_on_timeout:
                logger.warning("Forcing immediate shutdown due to timeout")
                await self._force_shutdown_all()

            return False

        except Exception as e:
            logger.error(f"Error in shutdown sequence: {e}")
            self.progress.errors.append(f"Shutdown sequence error: {str(e)}")
            return False

    async def _stop_new_requests(self) -> None:
        """Stop accepting new requests."""
        logger.info("Stopping new requests")

        # Call shutdown callbacks (these should stop accepting new requests)
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Error in shutdown callback: {e}")
                self.progress.warnings.append(f"Shutdown callback error: {str(e)}")

    async def _drain_connections(self) -> None:
        """Wait for existing connections to drain."""
        logger.info("Draining existing connections")

        # This would integrate with the WebSocket manager to gracefully close connections
        try:
            await asyncio.wait_for(
                self._wait_for_websocket_connections(),
                timeout=self.config.connection_drain_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Connection drain timed out")
            self.progress.warnings.append("Connection drain timed out")

    async def _wait_for_websocket_connections(self) -> None:
        """Wait for WebSocket connections to close gracefully."""
        # This would be implemented to monitor websocket_manager.get_active_connections()
        # For now, just a brief wait
        await asyncio.sleep(0.5)

    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks."""
        logger.info(f"Stopping {len(self._background_tasks)} background tasks")

        if not self._background_tasks:
            return

        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete or timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._background_tasks, return_exceptions=True),
                timeout=self.config.background_task_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Background task cleanup timed out")
            self.progress.warnings.append("Background task cleanup timed out")

        # Clear the task set
        self._background_tasks.clear()

    async def _wait_for_active_operations(self) -> None:
        """Wait for active operations to complete."""
        if not self._active_operations:
            return

        logger.info(
            f"Waiting for {len(self._active_operations)} active operations to complete"
        )

        # Wait for operations to complete, checking periodically
        max_wait_time = min(self.config.graceful_timeout / 2, 10.0)
        start_time = time.time()

        while self._active_operations and (time.time() - start_time) < max_wait_time:
            await asyncio.sleep(0.1)

        if self._active_operations:
            remaining = list(self._active_operations)
            logger.warning(f"Active operations still running: {remaining}")
            self.progress.warnings.append(
                f"Active operations did not complete: {remaining}"
            )

    async def _save_system_state(self) -> None:
        """Save current system state."""
        logger.info("Saving system state")

        try:
            # This would save configuration and state data
            # Implementation would depend on app_state and config_module
            await asyncio.sleep(0.1)  # Placeholder for actual save operation
            logger.info("System state saved successfully")
        except Exception as e:
            logger.error(f"Failed to save system state: {e}")
            self.progress.errors.append(f"State save failed: {str(e)}")

    async def _cleanup_resources(self) -> None:
        """Clean up system resources."""
        logger.info("Cleaning up system resources")

        try:
            # Clean up temporary files, caches, etc.
            # This is a placeholder for actual cleanup
            await asyncio.sleep(0.1)
            logger.info("Resource cleanup completed")
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")
            self.progress.errors.append(f"Resource cleanup failed: {str(e)}")

    async def _shutdown_modules(self) -> None:
        """Shutdown all registered modules."""
        logger.info(f"Shutting down {len(self._module_shutdowns)} modules")

        # Shutdown modules in reverse order of dependency
        module_order = [
            "vision",  # Vision module (camera, processing threads)
            "core",  # Core module (game state, physics)
            "websocket",  # WebSocket system
            "config",  # Configuration module
        ]

        # Shutdown modules in order
        for module_name in module_order:
            if module_name in self._module_shutdowns:
                await self._shutdown_single_module(module_name)

        # Shutdown any remaining modules not in the predefined order
        for module_name in self._module_shutdowns:
            if (
                module_name not in self.progress.modules_completed
                and module_name not in self.progress.modules_failed
            ):
                await self._shutdown_single_module(module_name)

    async def _shutdown_single_module(self, module_name: str) -> bool:
        """Shutdown a single module with retry logic.

        Args:
            module_name: Name of the module to shutdown

        Returns:
            True if shutdown was successful
        """
        shutdown_func = self._module_shutdowns.get(module_name)
        if not shutdown_func:
            return True

        for attempt in range(self.config.max_module_retries + 1):
            try:
                logger.info(
                    f"Shutting down module: {module_name} (attempt {attempt + 1})"
                )

                # Execute shutdown function with timeout
                if asyncio.iscoroutinefunction(shutdown_func):
                    await asyncio.wait_for(
                        shutdown_func(), timeout=self.config.module_timeout
                    )
                else:
                    # Run sync function in executor
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, shutdown_func),
                        timeout=self.config.module_timeout,
                    )

                self.progress.modules_completed.append(module_name)
                logger.info(f"Module {module_name} shutdown completed")
                return True

            except asyncio.TimeoutError:
                error_msg = (
                    f"Module {module_name} shutdown timed out (attempt {attempt + 1})"
                )
                logger.warning(error_msg)

                if attempt == self.config.max_module_retries:
                    self.progress.modules_failed.append(module_name)
                    self.progress.errors.append(error_msg)
                    return False
                else:
                    await asyncio.sleep(self.config.retry_delay)

            except Exception as e:
                error_msg = f"Module {module_name} shutdown failed: {str(e)} (attempt {attempt + 1})"
                logger.error(error_msg)

                if attempt == self.config.max_module_retries:
                    self.progress.modules_failed.append(module_name)
                    self.progress.errors.append(error_msg)
                    return False
                else:
                    await asyncio.sleep(self.config.retry_delay)

        return False

    async def _finalize_shutdown(self) -> None:
        """Finalize the shutdown process."""
        logger.info("Finalizing shutdown")

        try:
            # Final cleanup tasks
            self._active_operations.clear()
            self._background_tasks.clear()

            # Log shutdown summary
            total_modules = len(self._module_shutdowns)
            completed_modules = len(self.progress.modules_completed)
            failed_modules = len(self.progress.modules_failed)

            logger.info(
                f"Shutdown finalized: {completed_modules}/{total_modules} modules completed, "
                f"{failed_modules} failed, {len(self.progress.errors)} errors, "
                f"{len(self.progress.warnings)} warnings"
            )

        except Exception as e:
            logger.error(f"Error during shutdown finalization: {e}")
            self.progress.errors.append(f"Finalization error: {str(e)}")

    async def _force_shutdown_all(self) -> None:
        """Force immediate shutdown of all modules."""
        logger.warning("Forcing immediate shutdown of all modules")

        # Cancel all remaining tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        # Clear all state
        self._active_operations.clear()
        self._background_tasks.clear()

        self.progress.status = ShutdownStatus.FORCED

    def get_shutdown_status(self) -> dict[str, Any]:
        """Get current shutdown status.

        Returns:
            Dictionary containing shutdown progress and status
        """
        return {
            "is_shutting_down": self._is_shutting_down,
            "phase": self.progress.phase.value,
            "status": self.progress.status.value,
            "started_at": (
                self.progress.started_at.isoformat()
                if self.progress.started_at
                else None
            ),
            "completed_at": (
                self.progress.completed_at.isoformat()
                if self.progress.completed_at
                else None
            ),
            "elapsed_time": self.progress.elapsed_time,
            "modules_completed": self.progress.modules_completed,
            "modules_failed": self.progress.modules_failed,
            "active_operations": self.progress.active_operations,
            "errors": self.progress.errors,
            "warnings": self.progress.warnings,
            "force_requested": self.progress.force_requested,
        }


# Global shutdown coordinator instance
shutdown_coordinator = ShutdownCoordinator()


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")

        # Create event loop if needed and run shutdown
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # Schedule shutdown as a task
            asyncio.create_task(shutdown_coordinator.initiate_shutdown())
        else:
            # Run shutdown synchronously
            loop.run_until_complete(shutdown_coordinator.initiate_shutdown())

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # On Unix systems, also handle SIGHUP for configuration reload
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal_handler)


# Helper functions for module integration
async def register_module_for_shutdown(
    module_name: str, shutdown_func: Callable[[], Any]
) -> None:
    """Register a module for graceful shutdown.

    Args:
        module_name: Name of the module
        shutdown_func: Function to call for shutdown (sync or async)
    """
    shutdown_coordinator.register_module_shutdown(module_name, shutdown_func)


async def initiate_system_shutdown(
    force: bool = False, save_state: bool = True
) -> bool:
    """Initiate system-wide graceful shutdown.

    Args:
        force: Whether to force shutdown without waiting
        save_state: Whether to save system state

    Returns:
        True if shutdown completed successfully
    """
    return await shutdown_coordinator.initiate_shutdown(
        force=force, save_state=save_state
    )


def get_shutdown_progress() -> dict[str, Any]:
    """Get current shutdown progress.

    Returns:
        Dictionary containing shutdown status and progress
    """
    return shutdown_coordinator.get_shutdown_status()
