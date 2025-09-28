"""API Integration Module.

This module provides the main integration layer between the FastAPI application
and the backend modules (Configuration, Core, and Vision). It implements:
- Dependency injection for backend modules
- Service layer abstraction
- Event subscription and handling
- Data transformation between backend and API models
- Caching and performance optimization
- Health checks and monitoring
- Error handling and recovery
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Any, Callable, Optional

# Backend module imports
from ..config import ConfigurationModule
from ..core import CoreModule, CoreModuleConfig
from ..vision import VisionModule

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for API integration."""

    # Module settings
    enable_config_module: bool = True
    enable_core_module: bool = True
    enable_vision_module: bool = True

    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_cache_size: int = 1000

    # Event settings
    enable_events: bool = True
    event_buffer_size: int = 100

    # Health check settings
    health_check_interval: int = 30  # seconds
    module_timeout: float = 5.0  # seconds

    # Monitoring settings
    enable_metrics: bool = True
    metrics_retention: int = 3600  # 1 hour

    # Error handling
    auto_recovery: bool = True
    max_retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class ServiceHealth:
    """Health status for a service."""

    name: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    last_check: float
    error_message: Optional[str] = None
    response_time: Optional[float] = None


@dataclass
class IntegrationMetrics:
    """Metrics for integration layer."""

    requests_processed: int = 0
    average_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors_count: int = 0
    events_processed: int = 0
    uptime: float = 0.0


class IntegrationError(Exception):
    """Base exception for integration errors."""

    pass


class ServiceUnavailableError(IntegrationError):
    """Raised when a backend service is unavailable."""

    pass


class APIIntegration:
    """Main API Integration class.

    Orchestrates communication between the FastAPI application and backend modules.
    Provides service layer abstraction, dependency injection, and cross-cutting concerns.
    """

    def __init__(self, config: Optional[IntegrationConfig] = None):
        """Initialize API Integration.

        Args:
            config: Integration configuration
        """
        self.config = config or IntegrationConfig()
        self.metrics = IntegrationMetrics()
        self._start_time = time.time()

        # Module instances
        self.config_module: Optional[ConfigurationModule] = None
        self.core_module: Optional[CoreModule] = None
        self.vision_module: Optional[VisionModule] = None

        # Service layer
        self.game_service: Optional[GameStateService] = None
        self.config_service: Optional[ConfigurationService] = None
        self.calibration_service: Optional[CalibrationService] = None
        self.detection_service: Optional[DetectionService] = None

        # Internal state
        self._cache: dict[str, Any] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.event_buffer_size
        )
        self._event_handlers: dict[str, list[Callable]] = {}
        self._health_status: dict[str, ServiceHealth] = {}
        self._locks = {
            "cache": asyncio.Lock(),
            "health": asyncio.Lock(),
            "metrics": asyncio.Lock(),
        }

        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._event_processor_task: Optional[asyncio.Task] = None

        logger.info("API Integration initialized")

    async def startup(self) -> None:
        """Startup sequence for the integration layer.

        Initializes all backend modules and starts background tasks
        """
        try:
            logger.info("Starting API Integration...")

            # Initialize backend modules
            await self._initialize_modules()

            # Initialize service layer
            await self._initialize_services()

            # Setup event subscriptions
            await self._setup_event_subscriptions()

            # Start background tasks
            await self._start_background_tasks()

            logger.info("API Integration started successfully")

        except Exception as e:
            logger.error(f"Failed to start API Integration: {e}")
            await self.shutdown()
            raise IntegrationError(f"Startup failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown sequence for the integration layer.

        Gracefully stops all modules and background tasks
        """
        try:
            logger.info("Shutting down API Integration...")

            # Stop background tasks
            await self._stop_background_tasks()

            # Shutdown modules
            await self._shutdown_modules()

            # Clear caches and state
            await self._cleanup_state()

            logger.info("API Integration shutdown completed")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    # Module Management

    async def _initialize_modules(self) -> None:
        """Initialize backend modules with dependency injection."""
        try:
            # Initialize Configuration Module
            if self.config.enable_config_module:
                logger.info("Initializing Configuration Module...")
                self.config_module = ConfigurationModule()
                await self._update_health_status("config", "healthy")

            # Initialize Core Module
            if self.config.enable_core_module:
                logger.info("Initializing Core Module...")

                # Get core config from config module if available
                core_config = CoreModuleConfig()
                if self.config_module:
                    core_settings = await self._get_core_config_from_settings()
                    if core_settings:
                        core_config = CoreModuleConfig(**core_settings)

                self.core_module = CoreModule(core_config)
                await self._update_health_status("core", "healthy")

            # Initialize Vision Module
            if self.config.enable_vision_module:
                logger.info("Initializing Vision Module...")

                # Get vision config from config module if available
                vision_config = {}
                if self.config_module:
                    vision_settings = await self._get_vision_config_from_settings()
                    if vision_settings:
                        vision_config = vision_settings

                self.vision_module = VisionModule(vision_config)
                await self._update_health_status("vision", "healthy")

            logger.info("All modules initialized successfully")

        except Exception as e:
            logger.error(f"Module initialization failed: {e}")
            raise IntegrationError(f"Module initialization failed: {e}")

    async def _initialize_services(self) -> None:
        """Initialize service layer."""
        try:
            # Game State Service
            if self.core_module:
                self.game_service = GameStateService(self.core_module, self)

            # Configuration Service
            if self.config_module:
                self.config_service = ConfigurationService(self.config_module, self)

            # Calibration Service
            if self.vision_module:
                self.calibration_service = CalibrationService(self.vision_module, self)

            # Detection Service
            if self.vision_module:
                self.detection_service = DetectionService(self.vision_module, self)

            logger.info("Service layer initialized successfully")

        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            raise IntegrationError(f"Service initialization failed: {e}")

    async def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions from backend modules."""
        try:
            # Subscribe to Core Module events
            if self.core_module:
                self.core_module.subscribe_to_events(
                    "state_updated", self._handle_state_update
                )
                self.core_module.subscribe_to_events(
                    "game_reset", self._handle_game_reset
                )

            # Subscribe to Vision Module events
            if self.vision_module:
                self.vision_module.subscribe_to_events(
                    "detection_complete", self._handle_detection_complete
                )
                self.vision_module.subscribe_to_events(
                    "error_occurred", self._handle_vision_error
                )

            # Subscribe to Configuration Module events
            if self.config_module:
                # Configuration modules typically don't have async events,
                # but we can set up change notifications if needed
                pass

            logger.info("Event subscriptions setup successfully")

        except Exception as e:
            logger.error(f"Event subscription setup failed: {e}")
            raise IntegrationError(f"Event subscription setup failed: {e}")

    # Service Layer Implementation

    def get_game_service(self) -> "GameStateService":
        """Get game state service."""
        if not self.game_service:
            raise ServiceUnavailableError("Game service not available")
        return self.game_service

    def get_config_service(self) -> "ConfigurationService":
        """Get configuration service."""
        if not self.config_service:
            raise ServiceUnavailableError("Configuration service not available")
        return self.config_service

    def get_calibration_service(self) -> "CalibrationService":
        """Get calibration service."""
        if not self.calibration_service:
            raise ServiceUnavailableError("Calibration service not available")
        return self.calibration_service

    def get_detection_service(self) -> "DetectionService":
        """Get detection service."""
        if not self.detection_service:
            raise ServiceUnavailableError("Detection service not available")
        return self.detection_service

    # Caching Implementation

    async def get_cached(self, key: str) -> Optional[Any]:
        """Get cached value."""
        if not self.config.enable_caching:
            return None

        async with self._locks["cache"]:
            # Check if key exists and not expired
            if key in self._cache:
                timestamp = self._cache_timestamps.get(key, 0)
                if time.time() - timestamp < self.config.cache_ttl:
                    self.metrics.cache_hits += 1
                    return self._cache[key]
                else:
                    # Expired - remove from cache
                    del self._cache[key]
                    del self._cache_timestamps[key]

            self.metrics.cache_misses += 1
            return None

    async def set_cached(self, key: str, value: Any) -> None:
        """Set cached value."""
        if not self.config.enable_caching:
            return

        async with self._locks["cache"]:
            # Implement LRU eviction if cache is full
            if len(self._cache) >= self.config.max_cache_size:
                # Remove oldest entry
                oldest_key = min(
                    self._cache_timestamps.keys(),
                    key=lambda k: self._cache_timestamps[k],
                )
                del self._cache[oldest_key]
                del self._cache_timestamps[oldest_key]

            self._cache[key] = value
            self._cache_timestamps[key] = time.time()

    async def clear_cache(self, pattern: Optional[str] = None) -> None:
        """Clear cache entries."""
        async with self._locks["cache"]:
            if pattern is None:
                self._cache.clear()
                self._cache_timestamps.clear()
            else:
                # Remove keys matching pattern
                keys_to_remove = [k for k in self._cache if pattern in k]
                for key in keys_to_remove:
                    del self._cache[key]
                    del self._cache_timestamps[key]

    # Health Checks

    async def get_health_status(self) -> dict[str, Any]:
        """Get overall health status."""
        async with self._locks["health"]:
            overall_status = "healthy"
            unhealthy_services = []

            for service_name, health in self._health_status.items():
                if health.status == "unhealthy":
                    overall_status = "unhealthy"
                    unhealthy_services.append(service_name)
                elif health.status == "degraded" and overall_status == "healthy":
                    overall_status = "degraded"

            return {
                "status": overall_status,
                "timestamp": time.time(),
                "services": {
                    name: asdict(health) for name, health in self._health_status.items()
                },
                "unhealthy_services": unhealthy_services,
                "metrics": asdict(self.metrics),
            }

    async def check_service_health(self, service_name: str) -> ServiceHealth:
        """Check health of a specific service."""
        start_time = time.time()

        try:
            if service_name == "config" and self.config_module:
                # Test config module health
                await asyncio.wait_for(
                    self._check_config_health(), timeout=self.config.module_timeout
                )
                status = "healthy"
                error = None

            elif service_name == "core" and self.core_module:
                # Test core module health
                await asyncio.wait_for(
                    self._check_core_health(), timeout=self.config.module_timeout
                )
                status = "healthy"
                error = None

            elif service_name == "vision" and self.vision_module:
                # Test vision module health
                await asyncio.wait_for(
                    self._check_vision_health(), timeout=self.config.module_timeout
                )
                status = "healthy"
                error = None

            else:
                status = "unhealthy"
                error = f"Service {service_name} not available"

        except asyncio.TimeoutError:
            status = "unhealthy"
            error = f"Health check timeout for {service_name}"
        except Exception as e:
            status = "unhealthy"
            error = str(e)

        response_time = time.time() - start_time

        health = ServiceHealth(
            name=service_name,
            status=status,
            last_check=time.time(),
            error_message=error,
            response_time=response_time,
        )

        await self._update_health_status(service_name, status, error)
        return health

    # Event Handling

    async def _handle_state_update(self, event_data: dict[str, Any]) -> None:
        """Handle game state update events."""
        try:
            await self._queue_event("state_updated", event_data)
            # Clear relevant caches
            await self.clear_cache("game_state")
            await self.clear_cache("shot_analysis")
        except Exception as e:
            logger.error(f"Error handling state update: {e}")

    async def _handle_game_reset(self, event_data: dict[str, Any]) -> None:
        """Handle game reset events."""
        try:
            await self._queue_event("game_reset", event_data)
            # Clear all game-related caches
            await self.clear_cache("game_")
        except Exception as e:
            logger.error(f"Error handling game reset: {e}")

    async def _handle_detection_complete(self, event_data: dict[str, Any]) -> None:
        """Handle vision detection complete events."""
        try:
            await self._queue_event("detection_complete", event_data)
            # Update detection metrics
            if self.detection_service and "result" in event_data:
                await self.detection_service._update_detection_metrics(
                    event_data["result"]
                )
        except Exception as e:
            logger.error(f"Error handling detection complete: {e}")

    async def _handle_vision_error(self, event_data: dict[str, Any]) -> None:
        """Handle vision module errors."""
        try:
            await self._queue_event("vision_error", event_data)
            await self._update_health_status(
                "vision", "degraded", event_data.get("error")
            )
        except Exception as e:
            logger.error(f"Error handling vision error: {e}")

    # Background Tasks

    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        try:
            if self.config.enable_metrics:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
                self._event_processor_task = asyncio.create_task(
                    self._event_processor_loop()
                )

            logger.info("Background tasks started")

        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
            raise

    async def _stop_background_tasks(self) -> None:
        """Stop background tasks."""
        tasks = []

        if self._health_check_task:
            self._health_check_task.cancel()
            tasks.append(self._health_check_task)

        if self._event_processor_task:
            self._event_processor_task.cancel()
            tasks.append(self._event_processor_task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Background tasks stopped")

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                # Check all services
                for service_name in ["config", "core", "vision"]:
                    await self.check_service_health(service_name)

                await asyncio.sleep(self.config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _event_processor_loop(self) -> None:
        """Background event processor loop."""
        while True:
            try:
                # Process events from queue
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                    await self._process_event(event)
                    self.metrics.events_processed += 1
                except asyncio.TimeoutError:
                    continue  # No events to process

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processor loop: {e}")
                await asyncio.sleep(0.1)

    # Helper Methods

    async def _queue_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Queue an event for processing."""
        try:
            event = {
                "type": event_type,
                "data": data,
                "timestamp": time.time(),
                "id": str(uuid.uuid4()),
            }
            await self._event_queue.put(event)
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event: {event_type}")

    async def _process_event(self, event: dict[str, Any]) -> None:
        """Process a queued event."""
        try:
            event_type = event["type"]
            if event_type in self._event_handlers:
                for handler in self._event_handlers[event_type]:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Error in event handler for {event_type}: {e}")
        except Exception as e:
            logger.error(f"Error processing event: {e}")

    async def _update_health_status(
        self, service: str, status: str, error: Optional[str] = None
    ) -> None:
        """Update health status for a service."""
        async with self._locks["health"]:
            self._health_status[service] = ServiceHealth(
                name=service, status=status, last_check=time.time(), error_message=error
            )

    async def _get_core_config_from_settings(self) -> Optional[dict[str, Any]]:
        """Get core module configuration from settings."""
        try:
            if self.config_module:
                # Get core-specific settings
                core_config = {}

                # Get physics settings
                physics_enabled = self.config_module.get("core.physics.enabled", True)
                prediction_enabled = self.config_module.get(
                    "core.prediction.enabled", True
                )
                assistance_enabled = self.config_module.get(
                    "core.assistance.enabled", True
                )

                core_config.update(
                    {
                        "physics_enabled": physics_enabled,
                        "prediction_enabled": prediction_enabled,
                        "assistance_enabled": assistance_enabled,
                    }
                )

                return core_config
        except Exception as e:
            logger.warning(f"Failed to get core config from settings: {e}")

        return None

    async def _get_vision_config_from_settings(self) -> Optional[dict[str, Any]]:
        """Get vision module configuration from settings."""
        try:
            if self.config_module:
                # Get vision-specific settings
                vision_config = {}

                # Get camera settings
                camera_device = self.config_module.get("vision.camera.device_id", 0)
                camera_resolution = self.config_module.get(
                    "vision.camera.resolution", [1920, 1080]
                )
                camera_fps = self.config_module.get("vision.camera.fps", 30)

                vision_config.update(
                    {
                        "camera_device_id": camera_device,
                        "camera_resolution": tuple(camera_resolution),
                        "camera_fps": camera_fps,
                    }
                )

                return vision_config
        except Exception as e:
            logger.warning(f"Failed to get vision config from settings: {e}")

        return None

    async def _check_config_health(self) -> None:
        """Check configuration module health."""
        if self.config_module:
            # Test basic config operations
            test_key = "health.check.test"
            test_value = str(time.time())
            self.config_module.set(test_key, test_value)
            retrieved = self.config_module.get(test_key)
            if retrieved != test_value:
                raise Exception("Config read/write test failed")

    async def _check_core_health(self) -> None:
        """Check core module health."""
        if self.core_module:
            # Test core module operations
            self.core_module.get_current_state()
            metrics = self.core_module.get_performance_metrics()
            # Basic validation that module is responsive
            if metrics is None:
                raise Exception("Core module not responding")

    async def _check_vision_health(self) -> None:
        """Check vision module health."""
        if self.vision_module:
            # Test vision module operations
            stats = self.vision_module.get_statistics()
            if stats is None:
                raise Exception("Vision module not responding")

    async def _shutdown_modules(self) -> None:
        """Shutdown all modules."""
        try:
            # Stop vision module first (camera resources)
            if self.vision_module:
                self.vision_module.stop_capture()

            # No explicit shutdown needed for config and core modules

        except Exception as e:
            logger.error(f"Error shutting down modules: {e}")

    async def _cleanup_state(self) -> None:
        """Clean up internal state."""
        try:
            # Clear caches
            await self.clear_cache()

            # Clear event queue
            while not self._event_queue.empty():
                try:
                    self._event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        except Exception as e:
            logger.error(f"Error cleaning up state: {e}")


# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor API performance."""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = await func(self, *args, **kwargs)
            self.metrics.requests_processed += 1

            # Update average response time
            response_time = time.time() - start_time
            total_time = self.metrics.average_response_time * (
                self.metrics.requests_processed - 1
            )
            self.metrics.average_response_time = (
                total_time + response_time
            ) / self.metrics.requests_processed

            return result
        except Exception:
            self.metrics.errors_count += 1
            raise

    return wrapper


# Service Layer Classes


class BaseService:
    """Base service class with common functionality."""

    def __init__(self, integration: APIIntegration):
        self.integration = integration
        self.logger = logging.getLogger(self.__class__.__name__)

    async def _cache_get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        return await self.integration.get_cached(key)

    async def _cache_set(self, key: str, value: Any) -> None:
        """Set cached value."""
        await self.integration.set_cached(key, value)


class GameStateService(BaseService):
    """Service for game state operations."""

    def __init__(self, core_module: CoreModule, integration: APIIntegration):
        super().__init__(integration)
        self.core_module = core_module

    @monitor_performance
    async def get_current_state(self) -> Optional[dict[str, Any]]:
        """Get current game state."""
        # Check cache first
        cached = await self._cache_get("game_state")
        if cached:
            return cached

        # Get from core module
        state = self.core_module.get_current_state()
        if state:
            state_dict = asdict(state)
            await self._cache_set("game_state", state_dict)
            return state_dict

        return None

    @monitor_performance
    async def update_state(self, detection_data: dict[str, Any]) -> dict[str, Any]:
        """Update game state with new detection data."""
        try:
            state = await self.core_module.update_state(detection_data)
            state_dict = asdict(state)

            # Update cache
            await self._cache_set("game_state", state_dict)

            return state_dict
        except Exception as e:
            self.logger.error(f"Failed to update game state: {e}")
            raise

    @monitor_performance
    async def analyze_shot(self, target_ball: Optional[str] = None) -> dict[str, Any]:
        """Analyze current shot."""
        cache_key = f'shot_analysis_{target_ball or "auto"}'

        # Check cache
        cached = await self._cache_get(cache_key)
        if cached:
            return cached

        try:
            analysis = await self.core_module.analyze_shot(target_ball)
            analysis_dict = asdict(analysis)

            # Cache result
            await self._cache_set(cache_key, analysis_dict)

            return analysis_dict
        except Exception as e:
            self.logger.error(f"Failed to analyze shot: {e}")
            raise

    @monitor_performance
    async def suggest_shots(
        self, difficulty_filter: Optional[float] = None
    ) -> list[dict[str, Any]]:
        """Get shot suggestions."""
        cache_key = f'shot_suggestions_{difficulty_filter or "none"}'

        # Check cache
        cached = await self._cache_get(cache_key)
        if cached:
            return cached

        try:
            suggestions = await self.core_module.suggest_shots(
                difficulty_filter=difficulty_filter
            )
            suggestions_list = [asdict(suggestion) for suggestion in suggestions]

            # Cache result
            await self._cache_set(cache_key, suggestions_list)

            return suggestions_list
        except Exception as e:
            self.logger.error(f"Failed to get shot suggestions: {e}")
            raise


class ConfigurationService(BaseService):
    """Service for configuration operations."""

    def __init__(self, config_module: ConfigurationModule, integration: APIIntegration):
        super().__init__(integration)
        self.config_module = config_module

    @monitor_performance
    async def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        try:
            return self.config_module.get(key, default)
        except Exception as e:
            self.logger.error(f"Failed to get config {key}: {e}")
            raise

    @monitor_performance
    async def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        try:
            self.config_module.set(key, value)
            # Clear related caches
            await self.integration.clear_cache(f"config_{key}")
        except Exception as e:
            self.logger.error(f"Failed to set config {key}: {e}")
            raise

    @monitor_performance
    async def get_all_configs(self) -> dict[str, Any]:
        """Get all configuration values."""
        # Check cache
        cached = await self._cache_get("all_configs")
        if cached:
            return cached

        try:
            all_configs = self.config_module.get_all()
            await self._cache_set("all_configs", all_configs)
            return all_configs
        except Exception as e:
            self.logger.error(f"Failed to get all configs: {e}")
            raise


class CalibrationService(BaseService):
    """Service for calibration operations."""

    def __init__(self, vision_module: VisionModule, integration: APIIntegration):
        super().__init__(integration)
        self.vision_module = vision_module

    @monitor_performance
    async def calibrate_camera(self) -> dict[str, Any]:
        """Perform camera calibration."""
        try:
            success = self.vision_module.calibrate_camera()
            return {"success": success, "timestamp": time.time()}
        except Exception as e:
            self.logger.error(f"Camera calibration failed: {e}")
            raise

    @monitor_performance
    async def calibrate_colors(
        self, sample_image: Optional[Any] = None
    ) -> dict[str, Any]:
        """Perform color calibration."""
        try:
            results = self.vision_module.calibrate_colors(sample_image)
            return {
                "success": bool(results),
                "results": results,
                "timestamp": time.time(),
            }
        except Exception as e:
            self.logger.error(f"Color calibration failed: {e}")
            raise

    @monitor_performance
    async def set_roi(self, corners: list[tuple[int, int]]) -> dict[str, Any]:
        """Set region of interest."""
        try:
            self.vision_module.set_roi(corners)
            return {"success": True, "corners": corners, "timestamp": time.time()}
        except Exception as e:
            self.logger.error(f"Failed to set ROI: {e}")
            raise


class DetectionService(BaseService):
    """Service for detection operations."""

    def __init__(self, vision_module: VisionModule, integration: APIIntegration):
        super().__init__(integration)
        self.vision_module = vision_module
        self._detection_metrics = {
            "total_detections": 0,
            "successful_detections": 0,
            "average_processing_time": 0.0,
        }

    @monitor_performance
    async def start_detection(self) -> dict[str, Any]:
        """Start vision detection."""
        try:
            success = self.vision_module.start_capture()
            return {"success": success, "timestamp": time.time()}
        except Exception as e:
            self.logger.error(f"Failed to start detection: {e}")
            raise

    @monitor_performance
    async def stop_detection(self) -> dict[str, Any]:
        """Stop vision detection."""
        try:
            self.vision_module.stop_capture()
            return {"success": True, "timestamp": time.time()}
        except Exception as e:
            self.logger.error(f"Failed to stop detection: {e}")
            raise

    @monitor_performance
    async def get_latest_detection(self) -> Optional[dict[str, Any]]:
        """Get latest detection results."""
        try:
            result = self.vision_module.process_frame()
            if result:
                return asdict(result)
            return None
        except Exception as e:
            self.logger.error(f"Failed to get detection: {e}")
            raise

    @monitor_performance
    async def get_detection_statistics(self) -> dict[str, Any]:
        """Get detection statistics."""
        try:
            stats = self.vision_module.get_statistics()
            stats.update(self._detection_metrics)
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get detection statistics: {e}")
            raise

    async def _update_detection_metrics(self, result: Any) -> None:
        """Update internal detection metrics."""
        try:
            self._detection_metrics["total_detections"] += 1
            if result:
                self._detection_metrics["successful_detections"] += 1

                # Update average processing time
                processing_time = getattr(result, "processing_time", 0)
                total_time = self._detection_metrics["average_processing_time"] * (
                    self._detection_metrics["total_detections"] - 1
                )
                self._detection_metrics["average_processing_time"] = (
                    total_time + processing_time
                ) / self._detection_metrics["total_detections"]
        except Exception as e:
            self.logger.error(f"Failed to update detection metrics: {e}")


# Integration context manager for FastAPI lifespan
@asynccontextmanager
async def integration_lifespan(config: Optional[IntegrationConfig] = None):
    """Context manager for API integration lifecycle."""
    integration = APIIntegration(config)
    try:
        await integration.startup()
        yield integration
    finally:
        await integration.shutdown()


# Export main classes
__all__ = [
    "APIIntegration",
    "IntegrationConfig",
    "IntegrationError",
    "ServiceUnavailableError",
    "GameStateService",
    "ConfigurationService",
    "CalibrationService",
    "DetectionService",
    "integration_lifespan",
    "monitor_performance",
]
