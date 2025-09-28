#!/usr/bin/env python3
"""Integration test for the graceful shutdown system.

This script tests the complete shutdown implementation including:
- Module registration and coordination
- Graceful timeout handling
- Status reporting
- Error handling and forced termination
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add the backend directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from api.shutdown import ShutdownConfig, ShutdownCoordinator

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestModuleSimulator:
    """Simulates a module with various shutdown scenarios."""

    def __init__(
        self, name: str, shutdown_time: float = 0.1, should_fail: bool = False
    ):
        self.name = name
        self.shutdown_time = shutdown_time
        self.should_fail = should_fail
        self.shutdown_called = False

    async def async_shutdown(self):
        """Async shutdown method."""
        logger.info(f"Starting {self.name} async shutdown")
        await asyncio.sleep(self.shutdown_time)

        if self.should_fail:
            self.shutdown_called = True
            raise Exception(f"{self.name} shutdown failed!")

        self.shutdown_called = True
        logger.info(f"{self.name} async shutdown completed")

    def sync_shutdown(self):
        """Sync shutdown method."""
        logger.info(f"Starting {self.name} sync shutdown")
        time.sleep(self.shutdown_time)

        if self.should_fail:
            self.shutdown_called = True
            raise Exception(f"{self.name} shutdown failed!")

        self.shutdown_called = True
        logger.info(f"{self.name} sync shutdown completed")


async def test_basic_shutdown():
    """Test basic shutdown functionality."""
    logger.info("=" * 60)
    logger.info("Testing basic shutdown functionality")
    logger.info("=" * 60)

    try:
        config = ShutdownConfig(graceful_timeout=10.0, module_timeout=2.0)
        coordinator = ShutdownCoordinator(config)

        # Create test modules
        modules = [
            TestModuleSimulator("test_module_1", 0.1),
            TestModuleSimulator("test_module_2", 0.2),
            TestModuleSimulator("test_module_3", 0.1),
        ]

        # Register modules
        for module in modules:
            coordinator.register_module_shutdown(module.name, module.async_shutdown)

        # Add some active operations
        coordinator.register_active_operation("operation_1")
        coordinator.register_active_operation("operation_2")

        # Check initial status using the local coordinator
        status = coordinator.get_shutdown_status()
        assert status["status"] == "not_started"
        assert status["active_operations"] == 2

        # Simulate operations completing during shutdown
        async def complete_operations():
            await asyncio.sleep(0.3)
            coordinator.unregister_active_operation("operation_1")
            await asyncio.sleep(0.2)
            coordinator.unregister_active_operation("operation_2")

        # Start operation completion
        asyncio.create_task(complete_operations())

        # Execute shutdown
        start_time = time.time()
        success = await coordinator.initiate_shutdown(force=False, save_state=True)
        elapsed = time.time() - start_time

        # Verify results
        final_status = coordinator.get_shutdown_status()

        logger.info(f"Shutdown completed in {elapsed:.2f}s")
        logger.info(f"Success: {success}")
        logger.info(f"Status: {final_status['status']}")
        logger.info(f"Modules completed: {final_status['modules_completed']}")
        logger.info(f"Errors: {final_status['errors']}")

        # Assertions
        try:
            assert success, f"Shutdown should have succeeded, got: {success}"
            assert (
                final_status["status"] == "completed"
            ), f"Expected completed status, got: {final_status['status']}"
            assert (
                len(final_status["modules_completed"]) == 3
            ), f"Expected 3 completed modules, got: {len(final_status['modules_completed'])}"
            assert (
                final_status["active_operations"] == 0
            ), f"Expected 0 active operations, got: {final_status['active_operations']}"
            assert all(
                module.shutdown_called for module in modules
            ), f"Not all modules had shutdown called: {[m.shutdown_called for m in modules]}"
        except AssertionError as e:
            logger.error(f"Assertion failed in basic shutdown test: {e}")
            logger.error(f"Final status details: {final_status}")
            raise

        logger.info("✓ Basic shutdown test passed!")
        return True

    except Exception as e:
        logger.error(f"Exception in basic shutdown test: {e}", exc_info=True)
        raise


async def test_shutdown_with_failures():
    """Test shutdown with module failures."""
    logger.info("=" * 60)
    logger.info("Testing shutdown with module failures")
    logger.info("=" * 60)

    config = ShutdownConfig(
        graceful_timeout=10.0, module_timeout=1.0, max_module_retries=1
    )
    coordinator = ShutdownCoordinator(config)

    # Create test modules - one will fail
    modules = [
        TestModuleSimulator("good_module_1", 0.1),
        TestModuleSimulator("failing_module", 0.1, should_fail=True),
        TestModuleSimulator("good_module_2", 0.1),
    ]

    # Register modules
    for module in modules:
        coordinator.register_module_shutdown(module.name, module.async_shutdown)

    # Execute shutdown
    start_time = time.time()
    success = await coordinator.initiate_shutdown(force=False, save_state=False)
    elapsed = time.time() - start_time

    # Verify results
    final_status = coordinator.get_shutdown_status()

    logger.info(f"Shutdown completed in {elapsed:.2f}s")
    logger.info(f"Success: {success}")
    logger.info(f"Status: {final_status['status']}")
    logger.info(f"Modules completed: {final_status['modules_completed']}")
    logger.info(f"Modules failed: {final_status['modules_failed']}")
    logger.info(f"Errors: {final_status['errors']}")

    # Assertions
    assert not success, "Shutdown should have failed due to module failure"
    assert final_status["status"] == "failed"
    assert "failing_module" in final_status["modules_failed"]
    assert len(final_status["errors"]) > 0
    assert all(module.shutdown_called for module in modules)

    logger.info("✓ Shutdown with failures test passed!")
    return True


async def test_forced_shutdown():
    """Test forced shutdown."""
    logger.info("=" * 60)
    logger.info("Testing forced shutdown")
    logger.info("=" * 60)

    config = ShutdownConfig(
        graceful_timeout=2.0, module_timeout=5.0  # Long timeout to test force behavior
    )
    coordinator = ShutdownCoordinator(config)

    # Create modules with long shutdown times
    modules = [
        TestModuleSimulator("slow_module_1", 0.1),  # This one should complete
        TestModuleSimulator("slow_module_2", 0.1),  # This one too
    ]

    # Register modules
    for module in modules:
        coordinator.register_module_shutdown(module.name, module.async_shutdown)

    # Add persistent active operations
    coordinator.register_active_operation("persistent_op_1")
    coordinator.register_active_operation("persistent_op_2")

    # Execute forced shutdown
    start_time = time.time()
    success = await coordinator.initiate_shutdown(force=True, save_state=False)
    elapsed = time.time() - start_time

    # Verify results
    final_status = coordinator.get_shutdown_status()

    logger.info(f"Forced shutdown completed in {elapsed:.2f}s")
    logger.info(f"Success: {success}")
    logger.info(f"Status: {final_status['status']}")
    logger.info(f"Force requested: {final_status['force_requested']}")

    # Assertions for forced shutdown
    assert success, "Forced shutdown should succeed"
    assert final_status["force_requested"], "Force flag should be set"
    assert elapsed < 3.0, "Forced shutdown should be quick"

    logger.info("✓ Forced shutdown test passed!")
    return True


async def test_timeout_handling():
    """Test shutdown timeout handling."""
    logger.info("=" * 60)
    logger.info("Testing shutdown timeout handling")
    logger.info("=" * 60)

    config = ShutdownConfig(
        graceful_timeout=1.0,  # Very short timeout
        module_timeout=0.5,
        max_module_retries=0,  # No retries for faster timeout test
        force_kill_on_timeout=True,
    )
    coordinator = ShutdownCoordinator(config)

    # Create module with long shutdown time
    slow_module = TestModuleSimulator("very_slow_module", 2.0)  # Longer than timeout
    coordinator.register_module_shutdown(slow_module.name, slow_module.async_shutdown)

    # Execute shutdown
    start_time = time.time()
    success = await coordinator.initiate_shutdown(force=False, save_state=False)
    elapsed = time.time() - start_time

    # Verify results
    final_status = coordinator.get_shutdown_status()

    logger.info(f"Timeout test completed in {elapsed:.2f}s")
    logger.info(f"Success: {success}")
    logger.info(f"Status: {final_status['status']}")
    logger.info(f"Warnings: {final_status['warnings']}")

    # Should handle timeout gracefully
    assert elapsed < 2.0, "Should timeout quickly"
    logger.info("✓ Timeout handling test passed!")
    return True


async def test_status_reporting():
    """Test shutdown status reporting throughout the process."""
    logger.info("=" * 60)
    logger.info("Testing shutdown status reporting")
    logger.info("=" * 60)

    config = ShutdownConfig(graceful_timeout=5.0)
    coordinator = ShutdownCoordinator(config)

    # Register a module
    module = TestModuleSimulator("status_test_module", 0.5)
    coordinator.register_module_shutdown(module.name, module.async_shutdown)

    # Add operations
    coordinator.register_active_operation("status_op")

    # Check initial status
    status = coordinator.get_shutdown_status()
    assert status["status"] == "not_started"
    assert status["phase"] == "initiated"

    # Start shutdown in background
    shutdown_task = asyncio.create_task(
        coordinator.initiate_shutdown(force=False, save_state=True)
    )

    # Monitor status changes
    await asyncio.sleep(0.1)  # Let shutdown start
    status = coordinator.get_shutdown_status()
    assert status["status"] == "in_progress"
    logger.info(f"Status during shutdown: {status['phase']}")

    # Complete the operation
    coordinator.unregister_active_operation("status_op")

    # Wait for completion
    await shutdown_task

    # Check final status
    final_status = coordinator.get_shutdown_status()
    assert final_status["status"] == "completed"
    assert final_status["phase"] == "completed"
    assert final_status["elapsed_time"] > 0

    logger.info(f"Final status: {final_status['status']}")
    logger.info(f"Final phase: {final_status['phase']}")
    logger.info(f"Elapsed time: {final_status['elapsed_time']:.2f}s")

    logger.info("✓ Status reporting test passed!")
    return True


async def main():
    """Run all shutdown tests."""
    logger.info("Starting graceful shutdown integration tests")

    tests = [
        test_basic_shutdown,
        test_shutdown_with_failures,
        test_forced_shutdown,
        test_timeout_handling,
        test_status_reporting,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} failed: {e}")
            failed += 1

    logger.info("=" * 60)
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    if failed == 0:
        logger.info("🎉 All shutdown integration tests passed!")
        return True
    else:
        logger.error(f"❌ {failed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
