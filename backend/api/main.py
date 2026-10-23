"""Retired API lifecycle compatibility reference and configuration helpers."""

import copy
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Optional, Protocol, cast

# This must run before importing websocket, health, vision, or route modules:
# several of those modules read configuration while they are imported.
from backend.config import Config

Config.bootstrap_from_environment()

from backend.config import config
from backend.perception.provider import (
    PerceptionProvider,
    PerceptionProviderError,
    require_api_provider,
)
from backend.perception.rust_game_session import (
    RustGameSessionAuthorityClient,
    RustGameSessionAuthoritySettings,
)
from backend.perception.rust_live_supervisor import RustLiveSettings, RustLiveSupervisor
from backend.perception.rust_replay_supervisor import (
    RustReplaySettings,
    RustReplaySupervisor,
)
from backend.perception.scene_state_adapter import SceneSnapshotStatePublisher
from backend.system.health_monitor import health_monitor

from .dependencies import ApplicationState, app_state
from .middleware.error_handler import ErrorHandlerConfig, setup_error_handling
from .middleware.logging import LoggingConfig, setup_logging_middleware
from .middleware.metrics import MetricsConfig, MetricsMiddleware
from .middleware.performance import PerformanceConfig, setup_performance_monitoring
from .middleware.tracing import TracingConfig, setup_tracing_middleware
from .shutdown import register_module_for_shutdown, setup_signal_handlers
from .websocket import (
    initialize_websocket_system,
    message_broadcaster,
    shutdown_websocket_system,
    websocket_handler,
    websocket_manager,
)
from .websocket.rate_limit import rate_limiter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _create_rust_game_session_authority(
    runtime_config: Any,
    perception_settings: Any,
) -> RustGameSessionAuthorityClient | None:
    """Create an optional metadata client without making capture depend on it."""
    project_root = getattr(
        perception_settings, "project_root", Path(__file__).resolve().parents[2]
    )
    try:
        session_settings = RustGameSessionAuthoritySettings.from_configuration(
            runtime_config,
            perception_executable=perception_settings.executable_path,
            project_root=project_root,
        )
        return RustGameSessionAuthorityClient(session_settings)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        # Ball localization is the primary Rust service. An uninstalled or
        # misconfigured secondary metadata command must not stop capture.
        logger.warning(
            "Rust game-session metadata is unavailable; scene localization "
            "continues without it: %s",
            error,
        )
        return None


class _HealthMonitorLifecycle(Protocol):
    """Typed surface for health-monitor operations used by the API lifecycle."""

    def register_components(
        self,
        core_module: Any,
        config_module: Any,
        websocket_manager: Any,
        vision_module: Any,
        integration_service: Any,
    ) -> None: ...

    async def stop_monitoring(self) -> None: ...


class _RuntimeConfiguration:
    """Read lifecycle settings from app overrides, then the saved config."""

    def __init__(self, base: Any, overrides: dict[str, Any]) -> None:
        self._base = base
        self._overrides = overrides

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self._overrides
        for component in key.split("."):
            if not isinstance(value, dict) or component not in value:
                return self._base.get(key, default)
            value = value[component]
        return copy.deepcopy(value)

    def get_all(self) -> dict[str, Any]:
        """Return the saved configuration overlaid with application overrides."""
        get_all = getattr(self._base, "get_all", None)
        if not callable(get_all):
            raise TypeError("runtime configuration base must provide get_all()")
        merged = get_all()

        def merge(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
            for key, value in updates.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    target[key] = merge(target[key], value)
                else:
                    target[key] = copy.deepcopy(value)
            return target

        return merge(merged, self._overrides)


def get_middleware_config(
    development_mode: bool = False, config: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Get middleware configuration based on environment and config.

    Args:
        development_mode: Whether running in development mode
        config: Configuration dictionary (uses defaults if None)

    Returns:
        Dictionary of middleware configurations
    """
    # Use config values or fallback to defaults
    if config and "api" in config and "middleware" in config["api"]:
        middleware_cfg = config["api"]["middleware"]
    else:
        middleware_cfg = {}

    # Error handler config
    error_cfg = middleware_cfg.get("error_handler", {})

    # Logging config
    logging_cfg = middleware_cfg.get("logging", {})
    excluded_logging_paths = logging_cfg.get(
        "excluded_paths", ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
    )
    excluded_headers = logging_cfg.get(
        "excluded_headers", ["authorization", "cookie", "x-api-key", "x-auth-token"]
    )

    # Metrics config
    metrics_cfg = middleware_cfg.get("metrics", {})

    # Tracing config
    tracing_cfg = middleware_cfg.get("tracing", {})
    sample_rate_dev = tracing_cfg.get("sample_rate_development", 1.0)
    sample_rate_prod = tracing_cfg.get("sample_rate_production", 0.1)
    excluded_tracing_paths = tracing_cfg.get("excluded_paths", ["/health", "/metrics"])

    # Performance config
    perf_cfg = middleware_cfg.get("performance", {})
    slow_threshold = perf_cfg.get("slow_request_threshold_ms", 1000.0)
    excluded_perf_paths = perf_cfg.get("excluded_paths", ["/health", "/metrics"])

    return {
        "error_handler": ErrorHandlerConfig(
            include_traceback=development_mode,
            log_errors=error_cfg.get("log_errors", True),
            sanitize_errors=not development_mode,
        ),
        "logging": LoggingConfig(
            enable_request_logging=logging_cfg.get("enable_request_logging", True),
            enable_response_logging=logging_cfg.get("enable_response_logging", True),
            log_level=logging_cfg.get("log_level", "INFO"),
            log_body=development_mode or logging_cfg.get("log_body", False),
            log_headers=development_mode or logging_cfg.get("log_headers", False),
            max_body_size=logging_cfg.get("max_body_size", 1024),
            excluded_paths=set(excluded_logging_paths),
            excluded_headers=set(excluded_headers),
            include_performance_metrics=logging_cfg.get(
                "include_performance_metrics", True
            ),
            slow_request_threshold=logging_cfg.get("slow_request_threshold", 1.0),
            max_metrics_storage=logging_cfg.get("max_metrics_storage", 1000),
            recent_metrics_window=logging_cfg.get("recent_metrics_window", 100),
        ),
        "metrics": metrics_cfg,
        "tracing": TracingConfig(
            enable_tracing=tracing_cfg.get("enable_tracing", True),
            enable_correlation_ids=tracing_cfg.get("enable_correlation_ids", True),
            sample_rate=sample_rate_dev if development_mode else sample_rate_prod,
            excluded_paths=excluded_tracing_paths,
        ),
        "performance": PerformanceConfig(
            enable_monitoring=perf_cfg.get("enable_monitoring", True),
            enable_system_metrics=perf_cfg.get("enable_system_metrics", True),
            enable_endpoint_stats=perf_cfg.get("enable_endpoint_stats", True),
            enable_slow_query_detection=perf_cfg.get(
                "enable_slow_query_detection", True
            ),
            slow_request_threshold_ms=slow_threshold,
            metrics_retention_minutes=perf_cfg.get("metrics_retention_minutes", 60),
            stats_window_size=perf_cfg.get("stats_window_size", 1000),
            system_metrics_interval_seconds=perf_cfg.get(
                "system_metrics_interval_seconds", 30
            ),
            memory_alert_threshold_mb=perf_cfg.get("memory_alert_threshold_mb", 512),
            cpu_alert_threshold_percent=perf_cfg.get(
                "cpu_alert_threshold_percent", 80.0
            ),
            response_time_percentiles=perf_cfg.get(
                "response_time_percentiles", [50, 90, 95, 99]
            ),
            excluded_paths=excluded_perf_paths,
            endpoint_stats_recent_durations_maxlen=perf_cfg.get(
                "endpoint_stats_recent_durations_maxlen", 100
            ),
            cpu_percent_history_maxlen=perf_cfg.get("cpu_percent_history_maxlen", 60),
            memory_usage_history_maxlen=perf_cfg.get("memory_usage_history_maxlen", 60),
        ),
    }


# Application state is now imported from dependencies module


async def _register_shutdown_functions(app_state: ApplicationState) -> None:
    """Register shutdown functions for all modules."""
    # Register WebSocket system shutdown
    await register_module_for_shutdown("websocket", shutdown_websocket_system)

    # Register core module shutdown (if needed)
    if app_state.core_module:

        async def shutdown_core() -> None:
            """Shutdown core module resources."""
            # Clear caches and stop any background processing
            if app_state.core_module and hasattr(
                app_state.core_module, "trajectory_cache"
            ):
                app_state.core_module.trajectory_cache.clear()
            if app_state.core_module and hasattr(
                app_state.core_module, "analysis_cache"
            ):
                app_state.core_module.analysis_cache.clear()
            if app_state.core_module and hasattr(
                app_state.core_module, "collision_cache"
            ):
                app_state.core_module.collision_cache.clear()

            # Wait for any pending operations
            if app_state.core_module and hasattr(app_state.core_module, "_state_lock"):
                async with app_state.core_module._state_lock:
                    pass  # Ensure no state updates are in progress

            logger.info("Core module shutdown completed")

        await register_module_for_shutdown("core", shutdown_core)

    # Register configuration module shutdown
    if app_state.config_module:

        async def shutdown_config() -> None:
            """Shutdown configuration module."""
            # Simple config has no shutdown needed
            logger.info("Configuration module shutdown completed")

        await register_module_for_shutdown("config", shutdown_config)

    # Register vision module shutdown (when available)
    if app_state.vision_module:

        async def shutdown_vision() -> None:
            """Shutdown vision module."""
            try:
                # Stop capture and processing
                if app_state.vision_module and hasattr(
                    app_state.vision_module, "stop_capture"
                ):
                    app_state.vision_module.stop_capture()

                # Clean up camera resources
                if app_state.vision_module and hasattr(
                    app_state.vision_module, "camera"
                ):
                    camera = app_state.vision_module.camera
                    if hasattr(camera, "stop_capture"):
                        camera.stop_capture()

                logger.info("Vision module shutdown completed")
            except Exception as e:
                logger.error(f"Error during vision shutdown: {e}")

        await register_module_for_shutdown("vision", shutdown_vision)


@asynccontextmanager
async def lifespan(_app: Any) -> AsyncGenerator[None, None]:
    """Application lifecycle management."""
    # Startup
    logger.info("Starting up Billiards Trainer API...")
    app_state.startup_time = time.time()
    app_state.is_healthy = False
    app_state.config_module = config
    app_state.core_module = None
    app_state.vision_module = None
    app_state.integration_service = None
    app_state.scene_state_publisher = None
    app_state.game_session_authority = None
    app_state.perception_runtime = None
    runtime_config = getattr(_app.state, "runtime_config", config)

    try:
        selected_provider = require_api_provider(runtime_config)
    except PerceptionProviderError as error:
        logger.error("Perception provider preflight failed: %s", error)
        raise
    app_state.perception_provider = selected_provider.value
    logger.info("Selected perception provider: %s", selected_provider.value)

    try:
        # Configuration is now available as a simple singleton
        logger.info("Configuration system initialized")

        # Keep the API available for diagnostics when Redis is down, but report
        # unready and reject every new WebSocket admission until enforcement works.
        if not await rate_limiter.initialize():
            logger.error(
                "Rate-limit dependency is unavailable; WebSocket admission is closed"
            )

        # Vision module will be initialized lazily on first camera access
        # This prevents startup from hanging if camera initialization is slow
        logger.info("Vision module will be initialized on first camera/stream access")
        app_state.vision_module = None

        # Initialize WebSocket components (use global instances)
        logger.info("Initializing WebSocket components...")
        app_state.websocket_manager = websocket_manager
        app_state.websocket_handler = websocket_handler
        app_state.message_broadcaster = message_broadcaster

        # Start WebSocket system services
        await initialize_websocket_system()

        if runtime_config.get("perception.mode", "live") == "live":
            # Rust owns the sole capture and localization publisher.
            publisher = SceneSnapshotStatePublisher(message_broadcaster)
            app_state.scene_state_publisher = publisher
            live_settings = RustLiveSettings.from_configuration(runtime_config)
            app_state.game_session_authority = _create_rust_game_session_authority(
                runtime_config, live_settings
            )
            live_supervisor = RustLiveSupervisor(live_settings, publisher)
            app_state.perception_runtime = live_supervisor
            live_status = await live_supervisor.start()
            if not live_status.ready:
                raise RuntimeError(
                    "Rust live startup completed without a healthy localization"
                )
            logger.info(
                "Rust live capture published its first localization; "
                "no secondary perception publisher was started"
            )
        else:
            # Recorded replay is the only publisher in this branch.
            publisher = SceneSnapshotStatePublisher(message_broadcaster)
            app_state.scene_state_publisher = publisher
            replay_settings = RustReplaySettings.from_configuration(runtime_config)
            app_state.game_session_authority = _create_rust_game_session_authority(
                runtime_config, replay_settings
            )
            replay_supervisor = RustReplaySupervisor(replay_settings, publisher)
            app_state.perception_runtime = replay_supervisor
            replay_status = await replay_supervisor.start()
            logger.info(
                "Rust recorded replay published its first localization; "
                "live camera, preview and predictions remain unavailable"
            )
            if not replay_status.first_observation_received:
                raise RuntimeError(
                    "Rust replay startup completed without a usable observation"
                )

        # Register components with health monitor
        logger.info("Registering components with health monitor...")
        cast("_HealthMonitorLifecycle", health_monitor).register_components(
            core_module=app_state.core_module,
            config_module=app_state.config_module,
            websocket_manager=app_state.websocket_manager,
            vision_module=app_state.vision_module,
            integration_service=app_state.integration_service,
        )

        # Get health monitor configuration
        check_interval = runtime_config.get("api.health_monitor.check_interval", 5.0)

        # Start health monitoring
        await health_monitor.start_monitoring(check_interval=check_interval)

        # Register module shutdown functions for graceful shutdown
        await _register_shutdown_functions(app_state)

        # Setup signal handlers for graceful shutdown
        setup_signal_handlers()

        # Mark as healthy
        app_state.is_healthy = True

        startup_time = time.time() - app_state.startup_time
        logger.info(
            f"Billiards Trainer API started successfully in {startup_time:.2f}s"
        )

        yield

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        app_state.is_healthy = False
        raise

    finally:
        # Shutdown
        logger.info("Shutting down Billiards Trainer API...")

        if app_state.perception_runtime is not None:
            logger.info("Stopping supervised Rust perception...")
            try:
                await app_state.perception_runtime.close()
            except Exception:
                logger.exception("Error stopping Rust replay provider")
        app_state.perception_runtime = None
        app_state.scene_state_publisher = None
        app_state.game_session_authority = None

        # Stop health monitoring
        await cast("_HealthMonitorLifecycle", health_monitor).stop_monitoring()

        # Shutdown WebSocket system
        await shutdown_websocket_system()

        # Close the optional shared rate-limit client after WebSocket cleanup.
        try:
            await rate_limiter.close()
        except Exception:
            logger.exception("Error closing WebSocket rate-limit store")

        app_state.is_healthy = False
        logger.info("Billiards Trainer API shutdown complete")


# Dependency injection functions are now in dependencies module
