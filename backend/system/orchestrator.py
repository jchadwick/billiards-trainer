"""System Orchestrator - Main system coordination and lifecycle management."""

import asyncio
import logging
import signal
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

# Import core modules (API will be imported dynamically to avoid circular imports)
try:
    from ..config import ConfigurationModule
    from ..core import CoreModule, CoreModuleConfig
    from ..core.integration import (
        APIInterfaceImpl,
        ConfigInterfaceImpl,
        CoreModuleIntegrator,
        ProjectorInterfaceImpl,
        VisionInterfaceImpl,
    )
    from ..projector import ProjectorModule
    from ..vision import VisionModule
except ImportError:
    # If running from backend directory directly
    from core import CoreModule, CoreModuleConfig
    from core.integration import (
        CoreModuleIntegrator,
        VisionInterfaceImpl,
        APIInterfaceImpl,
        ProjectorInterfaceImpl,
        ConfigInterfaceImpl,
    )
    from projector import ProjectorModule
    from vision import VisionModule

    from config import ConfigurationModule

from .health import HealthMonitor, HealthStatus
from .monitoring import AlertManager, PerformanceMonitor
from .recovery import RecoveryManager
from .utils import ProcessManager, ResourceMonitor

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System operational states."""

    OFFLINE = "offline"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class SystemConfig:
    """System orchestrator configuration."""

    # Environment settings
    environment: str = "development"
    debug_mode: bool = False

    # Module enablement
    enable_vision: bool = True
    enable_projector: bool = True
    enable_api: bool = True
    enable_core: bool = True

    # Monitoring settings
    health_check_interval: float = 30.0
    performance_monitoring: bool = True
    auto_recovery: bool = True

    # Resource limits
    max_memory_mb: int = 4096
    max_cpu_percent: float = 80.0
    max_disk_usage_percent: float = 90.0

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    structured_logging: bool = True

    # Graceful shutdown timeout
    shutdown_timeout: float = 30.0


@dataclass
class ModuleStatus:
    """Status of a system module."""

    name: str
    state: SystemState
    health: HealthStatus
    startup_time: Optional[float] = None
    last_error: Optional[str] = None
    restart_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)


class SystemOrchestrator:
    """Main system orchestrator coordinating all modules and services."""

    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize the system orchestrator.

        Args:
            config: System configuration
        """
        self.config = config or SystemConfig()
        self.state = SystemState.OFFLINE
        self.startup_time: Optional[float] = None

        # Module instances
        self.modules: dict[str, Any] = {}
        self.module_status: dict[str, ModuleStatus] = {}

        # System components
        self.health_monitor = HealthMonitor()
        self.performance_monitor = PerformanceMonitor()
        self.alert_manager = AlertManager()
        self.recovery_manager = RecoveryManager()
        self.resource_monitor = ResourceMonitor()
        self.process_manager = ProcessManager()

        # Module integration - will be initialized after core module startup
        self.module_integrator: Optional[CoreModuleIntegrator] = None

        # Event tracking
        self.event_callbacks: dict[str, list[Callable]] = {}
        self.shutdown_requested = False

        # Background tasks
        self.background_tasks: list[asyncio.Task] = []

        logger.info("System Orchestrator initialized")

    async def start(self) -> bool:
        """Start the complete system.

        Returns:
            True if system started successfully
        """
        try:
            logger.info("Starting Billiards Trainer System...")
            self.state = SystemState.STARTING
            self.startup_time = time.time()

            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()

            # Initialize system monitoring
            await self._initialize_monitoring()

            # Start modules in order
            success = await self._start_modules()

            if success:
                # Initialize module integration after all modules are started
                await self._initialize_module_integration()

                # Start background monitoring tasks
                await self._start_background_tasks()

                self.state = SystemState.RUNNING
                startup_duration = time.time() - self.startup_time
                logger.info(f"System started successfully in {startup_duration:.2f}s")

                # Emit startup event
                await self._emit_event(
                    "system_started",
                    {
                        "startup_time": startup_duration,
                        "modules": list(self.modules.keys()),
                    },
                )

                return True
            else:
                self.state = SystemState.ERROR
                logger.error("System startup failed")
                return False

        except Exception as e:
            self.state = SystemState.ERROR
            logger.error(f"System startup failed: {e}")
            return False

    async def stop(self) -> None:
        """Stop the complete system gracefully."""
        try:
            logger.info("Stopping Billiards Trainer System...")
            self.state = SystemState.STOPPING
            self.shutdown_requested = True

            # Stop background tasks
            await self._stop_background_tasks()

            # Stop modules in reverse order
            await self._stop_modules()

            # Stop monitoring
            await self._stop_monitoring()

            self.state = SystemState.OFFLINE
            logger.info("System stopped successfully")

            # Emit shutdown event
            await self._emit_event("system_stopped", {})

        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
            self.state = SystemState.ERROR

    async def restart(self) -> bool:
        """Restart the complete system.

        Returns:
            True if restart successful
        """
        logger.info("Restarting system...")
        await self.stop()
        await asyncio.sleep(2.0)  # Brief pause
        return await self.start()

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status.

        Returns:
            Dictionary containing complete system status
        """
        try:
            # Get module statuses
            modules = {}
            for name, status in self.module_status.items():
                modules[name] = {
                    "state": status.state.value,
                    "health": status.health.value,
                    "startup_time": status.startup_time,
                    "last_error": status.last_error,
                    "restart_count": status.restart_count,
                    "metrics": status.metrics,
                }

            # Get system health
            system_health = await self.health_monitor.get_system_health()

            # Get performance metrics
            performance = await self.performance_monitor.get_current_metrics()

            # Get resource usage
            resources = await self.resource_monitor.get_current_usage()

            return {
                "system": {
                    "state": self.state.value,
                    "uptime": (
                        time.time() - self.startup_time if self.startup_time else 0
                    ),
                    "startup_time": self.startup_time,
                    "config": asdict(self.config),
                },
                "modules": modules,
                "health": asdict(system_health),
                "performance": asdict(performance),
                "resources": asdict(resources),
                "alerts": await self.alert_manager.get_active_alerts(),
            }

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}

    async def restart_module(self, module_name: str) -> bool:
        """Restart a specific module.

        Args:
            module_name: Name of module to restart

        Returns:
            True if restart successful
        """
        try:
            logger.info(f"Restarting module: {module_name}")

            if module_name not in self.modules:
                logger.error(f"Module {module_name} not found")
                return False

            # Stop module
            await self._stop_module(module_name)

            # Brief pause
            await asyncio.sleep(1.0)

            # Start module
            success = await self._start_module(module_name)

            if success:
                self.module_status[module_name].restart_count += 1
                logger.info(f"Module {module_name} restarted successfully")
            else:
                logger.error(f"Failed to restart module {module_name}")

            return success

        except Exception as e:
            logger.error(f"Error restarting module {module_name}: {e}")
            return False

    def subscribe_to_events(self, event_type: str, callback: Callable) -> str:
        """Subscribe to system events.

        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event occurs

        Returns:
            Subscription ID
        """
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []

        self.event_callbacks[event_type].append(callback)
        subscription_id = f"{event_type}_{len(self.event_callbacks[event_type])}"

        logger.debug(f"Subscription added: {subscription_id}")
        return subscription_id

    # Private methods

    async def _initialize_monitoring(self) -> None:
        """Initialize system monitoring components."""
        try:
            # Start health monitoring
            await self.health_monitor.start()

            # Start performance monitoring
            await self.performance_monitor.start()

            # Configure alert manager
            await self.alert_manager.configure(
                {
                    "max_memory_mb": self.config.max_memory_mb,
                    "max_cpu_percent": self.config.max_cpu_percent,
                    "max_disk_usage_percent": self.config.max_disk_usage_percent,
                }
            )

            # Start resource monitoring
            await self.resource_monitor.start()

            logger.info("System monitoring initialized")

        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
            raise

    async def _start_modules(self) -> bool:
        """Start all enabled modules in dependency order."""
        try:
            # Module startup order (dependencies first)
            module_order = []

            if self.config.enable_core:
                module_order.append("config")
                module_order.append("core")

            if self.config.enable_vision:
                module_order.append("vision")

            if self.config.enable_projector:
                module_order.append("projector")

            if self.config.enable_api:
                module_order.append("api")

            # Start each module
            for module_name in module_order:
                success = await self._start_module(module_name)
                if not success:
                    logger.error(f"Failed to start module: {module_name}")
                    return False

            logger.info("All modules started successfully")
            return True

        except Exception as e:
            logger.error(f"Module startup failed: {e}")
            return False

    async def _start_module(self, module_name: str) -> bool:
        """Start a specific module.

        Args:
            module_name: Name of module to start

        Returns:
            True if module started successfully
        """
        try:
            start_time = time.time()
            logger.info(f"Starting module: {module_name}")

            # Initialize module status
            self.module_status[module_name] = ModuleStatus(
                name=module_name,
                state=SystemState.STARTING,
                health=HealthStatus.UNKNOWN,
            )

            # Start specific module
            if module_name == "config":
                module = ConfigurationModule()
                await module.initialize()
                self.modules[module_name] = module

            elif module_name == "core":
                # Get configuration
                config_module = self.modules.get("config")
                if config_module:
                    await config_module.get_configuration()
                    core_config = CoreModuleConfig(
                        physics_enabled=True,
                        prediction_enabled=True,
                        assistance_enabled=True,
                        async_processing=True,
                        debug_mode=self.config.debug_mode,
                    )
                else:
                    core_config = CoreModuleConfig(debug_mode=self.config.debug_mode)

                module = CoreModule(core_config)
                self.modules[module_name] = module

            elif module_name == "vision":
                vision_config = {
                    "debug_mode": self.config.debug_mode,
                    "enable_threading": True,
                    "target_fps": 30,
                }
                module = VisionModule(vision_config)
                self.modules[module_name] = module

            elif module_name == "projector":
                projector_config = {"debug_mode": self.config.debug_mode}
                module = ProjectorModule(projector_config)
                await module.initialize()
                self.modules[module_name] = module

            elif module_name == "api":
                # API is handled separately as it needs special startup
                return await self._start_api_module()

            else:
                logger.error(f"Unknown module: {module_name}")
                return False

            # Update module status
            startup_time = time.time() - start_time
            self.module_status[module_name].state = SystemState.RUNNING
            self.module_status[module_name].health = HealthStatus.HEALTHY
            self.module_status[module_name].startup_time = startup_time

            # Register module with health monitor
            await self.health_monitor.register_module(module_name, module)

            logger.info(f"Module {module_name} started in {startup_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to start module {module_name}: {e}")
            self.module_status[module_name].state = SystemState.ERROR
            self.module_status[module_name].health = HealthStatus.UNHEALTHY
            self.module_status[module_name].last_error = str(e)
            return False

    async def _start_api_module(self) -> bool:
        """Start the API module with special handling."""
        try:
            # Import API dynamically to avoid circular imports
            try:
                from ..api.main import create_app as create_api_app
            except ImportError:
                from api.main import create_app as create_api_app

            # Create API app with dependency injection
            app_config = {"development_mode": self.config.environment == "development"}

            api_app = create_api_app(app_config)

            # Store API app reference
            self.modules["api"] = api_app

            # Note: Actual server startup is handled externally
            # This just registers the app for dependency injection

            self.module_status["api"] = ModuleStatus(
                name="api",
                state=SystemState.RUNNING,
                health=HealthStatus.HEALTHY,
                startup_time=0.1,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to start API module: {e}")
            return False

    async def _stop_modules(self) -> None:
        """Stop all modules in reverse order."""
        try:
            # Stop in reverse order
            module_order = list(self.modules.keys())
            module_order.reverse()

            for module_name in module_order:
                await self._stop_module(module_name)

            logger.info("All modules stopped")

        except Exception as e:
            logger.error(f"Error stopping modules: {e}")

    async def _stop_module(self, module_name: str) -> None:
        """Stop a specific module.

        Args:
            module_name: Name of module to stop
        """
        try:
            if module_name not in self.modules:
                return

            logger.info(f"Stopping module: {module_name}")

            module = self.modules[module_name]

            # Module-specific shutdown
            if module_name == "vision" and hasattr(module, "stop_capture"):
                module.stop_capture()
            elif module_name == "projector" and hasattr(module, "shutdown"):
                await module.shutdown()

            # Remove from health monitoring
            await self.health_monitor.unregister_module(module_name)

            # Update status
            if module_name in self.module_status:
                self.module_status[module_name].state = SystemState.OFFLINE
                self.module_status[module_name].health = HealthStatus.UNKNOWN

            # Remove module
            del self.modules[module_name]

            logger.info(f"Module {module_name} stopped")

        except Exception as e:
            logger.error(f"Error stopping module {module_name}: {e}")

    async def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks."""
        try:
            # Health monitoring task
            health_task = asyncio.create_task(self._health_monitoring_loop())
            self.background_tasks.append(health_task)

            # Performance monitoring task
            perf_task = asyncio.create_task(self._performance_monitoring_loop())
            self.background_tasks.append(perf_task)

            # Auto-recovery task
            if self.config.auto_recovery:
                recovery_task = asyncio.create_task(self._auto_recovery_loop())
                self.background_tasks.append(recovery_task)

            logger.info("Background tasks started")

        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")

    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks."""
        try:
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()

            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)

            self.background_tasks.clear()
            logger.info("Background tasks stopped")

        except Exception as e:
            logger.error(f"Error stopping background tasks: {e}")

    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while not self.shutdown_requested:
            try:
                # Check system health
                system_health = await self.health_monitor.get_system_health()

                # Update system state based on health
                if system_health.overall_status == HealthStatus.HEALTHY:
                    if self.state != SystemState.RUNNING:
                        self.state = SystemState.RUNNING
                elif system_health.overall_status == HealthStatus.DEGRADED:
                    self.state = SystemState.DEGRADED
                elif system_health.overall_status == HealthStatus.UNHEALTHY:
                    self.state = SystemState.ERROR

                # Emit health status event
                await self._emit_event(
                    "health_status",
                    {"system_health": asdict(system_health), "timestamp": time.time()},
                )

                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5.0)

    async def _performance_monitoring_loop(self) -> None:
        """Background performance monitoring loop."""
        while not self.shutdown_requested:
            try:
                # Collect performance metrics
                metrics = await self.performance_monitor.collect_metrics()

                # Check for alerts
                await self.alert_manager.check_thresholds(metrics)

                # Emit performance metrics event
                await self._emit_event(
                    "performance_metrics",
                    {"metrics": asdict(metrics), "timestamp": time.time()},
                )

                await asyncio.sleep(10.0)  # Collect every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(5.0)

    async def _auto_recovery_loop(self) -> None:
        """Background auto-recovery loop."""
        while not self.shutdown_requested:
            try:
                # Check for modules that need recovery
                for module_name, status in self.module_status.items():
                    if status.health == HealthStatus.UNHEALTHY:
                        logger.warning(
                            f"Module {module_name} is unhealthy, attempting recovery"
                        )

                        # Attempt recovery
                        success = await self.restart_module(module_name)

                        if success:
                            logger.info(f"Module {module_name} recovered successfully")
                        else:
                            logger.error(f"Failed to recover module {module_name}")

                await asyncio.sleep(30.0)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-recovery loop: {e}")
                await asyncio.sleep(10.0)

    async def _stop_monitoring(self) -> None:
        """Stop system monitoring components."""
        try:
            await self.health_monitor.stop()
            await self.performance_monitor.stop()
            await self.resource_monitor.stop()

            logger.info("System monitoring stopped")

        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")

    async def _initialize_module_integration(self) -> None:
        """Initialize module integration interfaces after all modules are started."""
        try:
            logger.info("Initializing module integration interfaces...")

            # Get the core module which should contain the event manager and game state manager
            core_module = self.modules.get("core")
            if not core_module:
                logger.error("Core module not available for integration")
                return

            # Create CoreModuleIntegrator with event manager and game state manager from core
            self.module_integrator = CoreModuleIntegrator(
                event_manager=core_module.event_manager,
                game_state_manager=core_module.game_state_manager,
            )

            # Create and register interface implementations
            await self._register_integration_interfaces()

            logger.info("Module integration interfaces initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize module integration: {e}")
            raise

    async def _register_integration_interfaces(self) -> None:
        """Register concrete interface implementations with the module integrator."""
        if not self.module_integrator:
            logger.error("Module integrator not initialized")
            return

        core_module = self.modules.get("core")
        if not core_module:
            logger.error("Core module not available for interface registration")
            return

        try:
            # Register Vision interface if vision module is available
            if "vision" in self.modules:
                vision_module = self.modules["vision"]
                vision_interface = VisionInterfaceImpl(
                    core_module.event_manager, vision_module=vision_module
                )
                self.module_integrator.register_vision_interface(vision_interface)

                # Wire vision module detection callbacks to integration layer
                def on_detection_complete(data: dict) -> None:
                    """Callback for when vision module completes detection."""
                    try:
                        # Extract detection result and convert to expected format
                        result = data.get("result")
                        frame_number = data.get("frame_number", 0)

                        # Broadcast frame if message_broadcaster is available
                        # Get API module from self.modules at callback time
                        api_mod = self.modules.get("api")
                        if (
                            api_mod
                            and hasattr(api_mod, "message_broadcaster")
                            and hasattr(vision_module, "get_current_frame")
                        ):
                            try:
                                import asyncio

                                frame = vision_module.get_current_frame()
                                if frame is not None and api_mod.message_broadcaster:
                                    # Get frame dimensions
                                    height, width = (
                                        frame.shape[:2] if len(frame.shape) >= 2 else (0, 0)
                                    )
                                    if width > 0 and height > 0:
                                        # Schedule async frame broadcast
                                        try:
                                            loop = asyncio.get_event_loop()
                                            loop.create_task(
                                                api_mod.message_broadcaster.broadcast_frame(
                                                    image_data=frame,
                                                    width=width,
                                                    height=height,
                                                    quality=75,  # Moderate quality for performance
                                                )
                                            )
                                        except RuntimeError:
                                            # No event loop running, skip frame broadcast
                                            pass
                            except Exception as frame_error:
                                logger.debug(
                                    f"Could not broadcast frame: {frame_error}"
                                )  # Debug level to avoid spam

                        if result:
                            # Convert DetectionResult to dict format expected by integration layer
                            detection_data = {
                                "timestamp": (
                                    result.timestamp
                                    if hasattr(result, "timestamp")
                                    else 0.0
                                ),
                                "frame_number": frame_number,
                                "balls": (
                                    [
                                        {
                                            "id": ball.id if hasattr(ball, "id") else i,
                                            "position": {
                                                "x": ball.position[0],
                                                "y": ball.position[1],
                                            },
                                            "number": (
                                                ball.number
                                                if hasattr(ball, "number")
                                                else 0
                                            ),
                                            "type": (
                                                ball.type.value
                                                if hasattr(ball, "type")
                                                else "unknown"
                                            ),
                                            "confidence": (
                                                ball.confidence
                                                if hasattr(ball, "confidence")
                                                else 1.0
                                            ),
                                        }
                                        for i, ball in enumerate(result.balls)
                                    ]
                                    if hasattr(result, "balls")
                                    else []
                                ),
                                "table": (
                                    {
                                        "corners": (
                                            result.table.corners
                                            if hasattr(result, "table") and result.table
                                            else []
                                        ),
                                    }
                                    if hasattr(result, "table")
                                    else None
                                ),
                                "cue": (
                                    {
                                        "position": (
                                            result.cue.position
                                            if hasattr(result, "cue") and result.cue
                                            else None
                                        ),
                                        "angle": (
                                            result.cue.angle
                                            if hasattr(result, "cue") and result.cue
                                            else None
                                        ),
                                    }
                                    if hasattr(result, "cue")
                                    else None
                                ),
                            }

                            # Forward to integration layer
                            vision_interface.receive_detection_data(detection_data)
                    except Exception as e:
                        logger.error(f"Error in vision detection callback: {e}")

                # Subscribe to vision module events
                vision_module.subscribe_to_events(
                    "detection_complete", on_detection_complete
                )
                logger.info(
                    "Vision interface registered with vision module and callbacks wired"
                )

            # Register API interface if API module is available
            # Note: WebSocket manager and broadcaster will be injected when API module starts
            if "api" in self.modules:
                # Get websocket components from API app if available
                websocket_manager = None
                message_broadcaster = None
                api_module = self.modules.get("api")
                if api_module:
                    if hasattr(api_module, "websocket_manager"):
                        websocket_manager = api_module.websocket_manager
                    if hasattr(api_module, "message_broadcaster"):
                        message_broadcaster = api_module.message_broadcaster

                api_interface = APIInterfaceImpl(
                    core_module.event_manager,
                    websocket_manager=websocket_manager,
                    message_broadcaster=message_broadcaster,
                )
                self.module_integrator.register_api_interface(api_interface)
                logger.info(
                    f"API interface registered (websocket_manager: {websocket_manager is not None}, "
                    f"message_broadcaster: {message_broadcaster is not None})"
                )

            # Register Projector interface if projector module is available
            if "projector" in self.modules:
                projector_module = self.modules["projector"]
                projector_interface = ProjectorInterfaceImpl(
                    core_module.event_manager, projector_module=projector_module
                )
                self.module_integrator.register_projector_interface(projector_interface)
                logger.info("Projector interface registered with projector module")

            # Register Config interface if config module is available
            if "config" in self.modules:
                config_module = self.modules["config"]
                config_interface = ConfigInterfaceImpl(
                    core_module.event_manager, config_module=config_module
                )
                self.module_integrator.register_config_interface(config_interface)
                logger.info("Config interface registered with config module")

        except Exception as e:
            logger.error(f"Failed to register integration interfaces: {e}")
            raise

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True

            # Create shutdown task
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.stop())

        # Register signal handlers
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        if hasattr(signal, "SIGHUP"):  # Unix only
            signal.signal(signal.SIGHUP, signal_handler)

    async def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit system event to subscribers.

        Args:
            event_type: Type of event
            data: Event data
        """
        try:
            if event_type in self.event_callbacks:
                for callback in self.event_callbacks[event_type]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
                    except Exception as e:
                        logger.warning(f"Error in event callback for {event_type}: {e}")

        except Exception as e:
            logger.error(f"Error emitting event {event_type}: {e}")
