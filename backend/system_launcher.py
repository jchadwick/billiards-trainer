#!/usr/bin/env python3
"""Main system launcher for the Billiards Trainer system.

This script provides the primary entry point for starting the complete
billiards trainer system including all modules and monitoring.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import uvicorn

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from system import create_orchestrator
from system.recovery import RecoveryAction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("system.log")],
)

logger = logging.getLogger(__name__)


class SystemLauncher:
    """Main system launcher and coordinator."""

    def __init__(self, config):
        """Initialize system launcher.

        Args:
            config: System configuration
        """
        self.config = config
        self.orchestrator = None
        self.api_server: uvicorn.Server = None

    async def start_system(self) -> bool:
        """Start the complete system.

        Returns:
            True if system started successfully
        """
        try:
            logger.info("=== Starting Billiards Trainer System ===")

            # Create system orchestrator
            self.orchestrator = create_orchestrator(self.config)

            # Register recovery callbacks
            self._setup_recovery_callbacks()

            # Start the orchestrator
            success = await self.orchestrator.start()

            if not success:
                logger.error("Failed to start system orchestrator")
                return False

            # Start API server if enabled
            if self.config.enable_api:
                await self._start_api_server()

            logger.info("=== System Started Successfully ===")
            return True

        except Exception as e:
            logger.error(f"System startup failed: {e}")
            return False

    async def stop_system(self) -> None:
        """Stop the complete system gracefully."""
        try:
            logger.info("=== Stopping Billiards Trainer System ===")

            # Stop API server
            if self.api_server:
                await self._stop_api_server()

            # Stop orchestrator
            if self.orchestrator:
                await self.orchestrator.stop()

            logger.info("=== System Stopped Successfully ===")

        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")

    async def run_system(self) -> None:
        """Run the system until shutdown is requested."""
        try:
            # Start system
            success = await self.start_system()

            if not success:
                logger.error("Failed to start system")
                return

            # Wait for shutdown signal
            logger.info("System running - Press Ctrl+C to stop")

            try:
                # Keep running until interrupted
                while not self.orchestrator.shutdown_requested:
                    await asyncio.sleep(1.0)

            except KeyboardInterrupt:
                logger.info("Shutdown requested by user")

        except Exception as e:
            logger.error(f"Error running system: {e}")

        finally:
            # Always attempt cleanup
            await self.stop_system()

    async def _start_api_server(self) -> None:
        """Start the API server."""
        try:
            logger.info("Starting API server...")

            # Get the API app from orchestrator
            api_app = self.orchestrator.modules.get("api")

            if not api_app:
                logger.error("API app not available")
                return

            # Configure uvicorn
            config = uvicorn.Config(
                app=api_app,
                host=self.config.api_host,
                port=self.config.api_port,
                workers=self.config.api_workers,
                log_level=self.config.log_level.lower(),
                access_log=True,
                use_colors=True,
            )

            self.api_server = uvicorn.Server(config)

            # Start server in background
            asyncio.create_task(self.api_server.serve())

            logger.info(
                f"API server started on {self.config.api_host}:{self.config.api_port}"
            )

        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            raise

    async def _stop_api_server(self) -> None:
        """Stop the API server."""
        try:
            if self.api_server:
                logger.info("Stopping API server...")
                self.api_server.should_exit = True
                await asyncio.sleep(1.0)  # Give it time to shutdown
                logger.info("API server stopped")

        except Exception as e:
            logger.error(f"Error stopping API server: {e}")

    def _setup_recovery_callbacks(self) -> None:
        """Setup recovery callbacks for the orchestrator."""
        recovery_manager = self.orchestrator.recovery_manager

        # Register recovery callbacks
        recovery_manager.register_callback(
            RecoveryAction.RESTART_MODULE, self._restart_module_callback
        )

        recovery_manager.register_callback(
            RecoveryAction.RESTART_SYSTEM, self._restart_system_callback
        )

        recovery_manager.register_callback(
            RecoveryAction.RESET_STATE, self._reset_state_callback
        )

        recovery_manager.register_callback(
            RecoveryAction.CLEAR_CACHE, self._clear_cache_callback
        )

        logger.info("Recovery callbacks configured")

    async def _restart_module_callback(self, module_name: str) -> bool:
        """Callback for module restart recovery.

        Args:
            module_name: Name of module to restart

        Returns:
            True if restart was successful
        """
        try:
            logger.info(f"Recovery: Restarting module {module_name}")
            return await self.orchestrator.restart_module(module_name)

        except Exception as e:
            logger.error(f"Recovery callback failed for module {module_name}: {e}")
            return False

    async def _restart_system_callback(self) -> bool:
        """Callback for system restart recovery.

        Returns:
            True if restart was successful
        """
        try:
            logger.warning("Recovery: Restarting entire system")
            return await self.orchestrator.restart()

        except Exception as e:
            logger.error(f"System restart recovery failed: {e}")
            return False

    async def _reset_state_callback(self, module_name: str) -> bool:
        """Callback for state reset recovery.

        Args:
            module_name: Name of module to reset

        Returns:
            True if reset was successful
        """
        try:
            logger.info(f"Recovery: Resetting state for module {module_name}")

            # Get module instance
            module = self.orchestrator.modules.get(module_name)
            if not module:
                return False

            # Module-specific state reset
            if module_name == "core" and hasattr(module, "reset_game"):
                await module.reset_game()
                return True
            elif module_name == "vision" and hasattr(module, "stop_capture"):
                # Restart vision capture
                module.stop_capture()
                await asyncio.sleep(1.0)
                return module.start_capture()

            return True

        except Exception as e:
            logger.error(f"State reset recovery failed for module {module_name}: {e}")
            return False

    async def _clear_cache_callback(self, module_name: str) -> bool:
        """Callback for cache clearing recovery.

        Args:
            module_name: Name of module to clear cache for

        Returns:
            True if cache clearing was successful
        """
        try:
            logger.info(f"Recovery: Clearing cache for module {module_name}")

            # Get module instance
            module = self.orchestrator.modules.get(module_name)
            if not module:
                return True  # No module to clear cache for

            # Module-specific cache clearing
            if module_name == "core":
                if hasattr(module, "trajectory_cache"):
                    module.trajectory_cache.clear()
                if hasattr(module, "analysis_cache"):
                    module.analysis_cache.clear()
                if hasattr(module, "collision_cache"):
                    module.collision_cache.clear()

            return True

        except Exception as e:
            logger.error(
                f"Cache clearing recovery failed for module {module_name}: {e}"
            )
            return False


def create_config_from_args(args):
    """Create system configuration from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        System configuration dictionary
    """
    return {
        "environment": args.environment,
        "debug_mode": args.debug,
        "enable_vision": not args.no_vision,
        "enable_projector": not args.no_projector,
        "enable_api": not args.no_api,
        "enable_core": not args.no_core,
        "health_check_interval": args.health_interval,
        "performance_monitoring": not args.no_monitoring,
        "auto_recovery": not args.no_recovery,
        "api_host": args.host,
        "api_port": args.port,
        "api_workers": args.workers,
        "log_level": args.log_level,
        "log_file": args.log_file,
        "shutdown_timeout": args.shutdown_timeout,
    }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Billiards Trainer System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start full system in development mode
  python system_launcher.py --environment development

  # Start production system
  python system_launcher.py --environment production --host 0.0.0.0 --port 8000

  # Start with specific modules disabled
  python system_launcher.py --no-projector --no-vision

  # Start with debug logging
  python system_launcher.py --debug --log-level DEBUG
        """,
    )

    # Environment settings
    parser.add_argument(
        "--environment",
        choices=["development", "production", "testing"],
        default="development",
        help="Environment mode (default: development)",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Module control
    parser.add_argument(
        "--no-vision", action="store_true", help="Disable vision module"
    )

    parser.add_argument(
        "--no-projector", action="store_true", help="Disable projector module"
    )

    parser.add_argument("--no-api", action="store_true", help="Disable API module")

    parser.add_argument("--no-core", action="store_true", help="Disable core module")

    # Monitoring settings
    parser.add_argument(
        "--no-monitoring", action="store_true", help="Disable performance monitoring"
    )

    parser.add_argument(
        "--no-recovery", action="store_true", help="Disable auto-recovery"
    )

    parser.add_argument(
        "--health-interval",
        type=float,
        default=30.0,
        help="Health check interval in seconds (default: 30)",
    )

    # API server settings
    parser.add_argument(
        "--host", default="0.0.0.0", help="API server host (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port", type=int, default=8000, help="API server port (default: 8000)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of API server workers (default: 1)",
    )

    # Logging settings
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument("--log-file", help="Log file path (default: stdout only)")

    # System settings
    parser.add_argument(
        "--shutdown-timeout",
        type=float,
        default=30.0,
        help="Graceful shutdown timeout in seconds (default: 30)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Create configuration
    config = create_config_from_args(args)

    # Update logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create and run launcher
    launcher = SystemLauncher(config)

    try:
        await launcher.run_system()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
