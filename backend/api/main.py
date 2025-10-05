"""FastAPI application setup and configuration."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.websockets import WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

# Internal imports
# Import core module
try:
    from ..core import CoreModule, CoreModuleConfig
except ImportError:
    import sys
    from pathlib import Path

    backend_path = Path(__file__).parent.parent
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    from core import CoreModule, CoreModuleConfig  # type: ignore[import-not-found]

# Import config module - use absolute import to avoid conflict with api.routes.config
import sys
from pathlib import Path

backend_path = Path(__file__).parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))
from config.manager import ConfigurationModule  # type: ignore

from .dependencies import ApplicationState, app_state
from .middleware.error_handler import ErrorHandlerConfig, setup_error_handling
from .middleware.logging import LoggingConfig, setup_logging_middleware
from .middleware.metrics import MetricsMiddleware
from .middleware.performance import PerformanceConfig, setup_performance_monitoring
from .middleware.tracing import TracingConfig, setup_tracing_middleware
from .routes import (
    calibration,
    config,
    diagnostics,
    game,
    health,
    logs,
    modules,
    stream,
)
from .shutdown import register_module_for_shutdown, setup_signal_handlers
from .websocket import (
    initialize_websocket_system,
    message_broadcaster,
    shutdown_websocket_system,
    websocket_handler,
    websocket_manager,
)
from .websocket.endpoints import websocket_router

# Import health monitor with fallback
try:
    from ..system.health_monitor import health_monitor
except ImportError:
    try:
        from system.health_monitor import health_monitor  # type: ignore
    except ImportError:
        # Fallback: create a minimal health monitor interface
        from typing import Any as _Any

        class MockHealthMonitor:
            """Mock health monitor for when system.health_monitor is not available."""

            def register_components(self, **kwargs: _Any) -> None:
                """Register components for monitoring."""
                pass

            async def start_monitoring(self, check_interval: float = 5.0) -> None:
                """Start monitoring."""
                pass

            async def stop_monitoring(self) -> None:
                """Stop monitoring."""
                pass

        health_monitor = MockHealthMonitor()  # type: ignore[assignment]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_middleware_config(development_mode: bool = False) -> dict[str, Any]:
    """Get middleware configuration based on environment."""
    return {
        "error_handler": ErrorHandlerConfig(
            include_traceback=development_mode,
            log_errors=True,
            sanitize_errors=not development_mode,
        ),
        "logging": LoggingConfig(
            enable_request_logging=True,
            enable_response_logging=True,
            log_body=development_mode,
            log_headers=development_mode,
            excluded_paths={"/health", "/metrics", "/docs", "/redoc", "/openapi.json"},
        ),
        "tracing": TracingConfig(
            enable_tracing=True,
            enable_correlation_ids=True,
            sample_rate=1.0 if development_mode else 0.1,
            excluded_paths=["/health", "/metrics"],
        ),
        "performance": PerformanceConfig(
            enable_monitoring=True,
            enable_system_metrics=True,
            slow_request_threshold_ms=500.0,
            excluded_paths=["/health", "/metrics"],
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
            # Save current configuration
            try:
                if app_state.config_module:
                    app_state.config_module.save_config()

                    # Stop file watchers if they exist
                    if (
                        hasattr(app_state.config_module, "_config_watcher")
                        and app_state.config_module._config_watcher
                    ):
                        watcher = app_state.config_module._config_watcher
                        if hasattr(watcher, "stop"):
                            watcher.stop()

                    # Create backup if configured (skipping due to signature issues)
                    # The backup method may have different signature than expected
                    # if hasattr(app_state.config_module, "_backup"):
                    #     backup = app_state.config_module._backup
                    #     if hasattr(backup, "create_backup"):
                    #         backup_metadata = backup.create_backup()

                logger.info("Configuration module shutdown completed")
            except Exception as e:
                logger.error(f"Error during configuration shutdown: {e}")

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
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle management."""
    # Startup
    logger.info("Starting up Billiards Trainer API...")
    app_state.startup_time = time.time()

    try:
        # Initialize configuration module
        logger.info("Initializing configuration module...")
        app_state.config_module = ConfigurationModule(enable_hot_reload=False)
        await app_state.config_module.initialize()

        # Get configuration
        config = await app_state.config_module.get_configuration()

        # Initialize core module
        logger.info("Initializing core module...")
        core_config = CoreModuleConfig(
            physics_enabled=True,
            prediction_enabled=True,
            assistance_enabled=True,
            async_processing=True,
            debug_mode=config.system.debug if hasattr(config, "system") else False,
        )
        app_state.core_module = CoreModule(core_config)

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

        # Register components with health monitor
        logger.info("Registering components with health monitor...")
        health_monitor.register_components(
            core_module=app_state.core_module,
            config_module=app_state.config_module,
            websocket_manager=app_state.websocket_manager,
        )

        # Start health monitoring
        await health_monitor.start_monitoring(check_interval=5.0)

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

        # Stop health monitoring
        await health_monitor.stop_monitoring()

        # Shutdown WebSocket system
        await shutdown_websocket_system()

        if app_state.core_module:
            # Core module doesn't have explicit shutdown, but we can clean up
            pass

        if app_state.config_module:
            # Configuration module cleanup if needed
            pass

        app_state.is_healthy = False
        logger.info("Billiards Trainer API shutdown complete")


def create_app(config_override: Optional[dict[str, Any]] = None) -> FastAPI:
    """Create and configure FastAPI application."""
    # Create FastAPI app with lifecycle management
    app = FastAPI(
        title="Billiards Trainer API",
        description="RESTful API and WebSocket server for billiards training system",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Apply configuration overrides if provided
    if config_override:
        app.extra = {"config_override": config_override}

    # Determine if running in development mode
    import os

    development_mode = (
        config_override.get("development_mode", False)
        if config_override
        else os.getenv("ENVIRONMENT", "production").lower() == "development"
        or os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    )

    # Get middleware configuration
    middleware_config = get_middleware_config(development_mode)

    # Add CORS middleware first - must be before other middleware for WebSocket support
    # In production, you should restrict origins to your frontend domains
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Restrict to specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add middleware in reverse order (last added = first executed)
    # This ensures proper execution order for request/response processing

    # 1. Health metrics tracking (last - measures everything)
    app.add_middleware(MetricsMiddleware)

    # 2. Performance monitoring (last - measures everything)
    setup_performance_monitoring(app, middleware_config["performance"])

    # 3. Request tracing and correlation IDs
    setup_tracing_middleware(app, middleware_config["tracing"])

    # 4. Request/response logging
    setup_logging_middleware(app, middleware_config["logging"])

    # 5. Error handling (handles errors from subsequent middleware)
    setup_error_handling(app, middleware_config["error_handler"])

    # Middleware setup complete

    # Include routers with API prefix
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(config.router, prefix="/api/v1")
    app.include_router(calibration.router, prefix="/api/v1/vision")
    app.include_router(game.router, prefix="/api/v1")
    app.include_router(stream.router, prefix="/api/v1")
    app.include_router(modules.router, prefix="/api/v1")
    app.include_router(diagnostics.router, prefix="/api/v1")
    app.include_router(logs.router, prefix="/api/v1")

    # Include WebSocket management endpoints
    app.include_router(websocket_router, prefix="/api/v1/websocket")

    # Include the actual WebSocket endpoint at root level (no prefix)
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """Main WebSocket endpoint with full handler integration.

        Handles:
        - Connection lifecycle (connect, message handling, disconnect)
        - Message subscriptions (frame, state, trajectory, alert, config)
        - Connection monitoring and health checks
        - Rate limiting and quality tracking
        """
        client_id: Optional[str] = None

        try:
            # Connect via WebSocketHandler - this accepts the connection and returns client_id
            logger.info(f"New WebSocket connection from {websocket.client}")
            client_id = await websocket_handler.connect(websocket)
            logger.info(f"WebSocket connected successfully: {client_id}")

            # Message loop - handle incoming messages
            while True:
                try:
                    # Wait for client message
                    message = await websocket.receive_text()

                    # Handle the message via WebSocketHandler
                    await websocket_handler.handle_message(client_id, message)

                except WebSocketDisconnect:
                    # Client disconnected normally - exit loop
                    logger.info(f"WebSocket client disconnected: {client_id}")
                    break
                except Exception as msg_error:
                    # Check if this is a disconnect-related error
                    error_msg = str(msg_error)
                    if "disconnect" in error_msg.lower() or "receive" in error_msg.lower():
                        logger.info(
                            f"WebSocket disconnected (via exception): {client_id}"
                        )
                        break

                    logger.error(
                        f"Error processing message from {client_id}: {msg_error}"
                    )
                    # Continue processing - don't disconnect on single message error

        except Exception as e:
            # Log connection error
            if client_id:
                logger.error(f"WebSocket error for {client_id}: {e}")
            else:
                logger.error(f"WebSocket connection error: {e}")

        finally:
            # Always disconnect and cleanup
            if client_id:
                logger.info(f"WebSocket disconnecting: {client_id}")
                await websocket_handler.disconnect(client_id)

            # Ensure WebSocket is closed
            with suppress(Exception):
                await websocket.close()

    app.add_websocket_route("/ws", websocket_endpoint)

    # API root endpoint
    @app.get("/api")
    async def api_root() -> dict[str, Any]:
        return {
            "message": "Billiards Trainer API",
            "version": "1.0.0",
            "status": "healthy" if app_state.is_healthy else "unhealthy",
            "docs": "/docs",
        }

    # Serve static files for the frontend with SPA fallback
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        # Mount static files but not at root to avoid conflicts
        app.mount(
            "/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets"
        )

        # Serve individual static files
        @app.get("/favicon.ico")
        async def favicon() -> FileResponse:
            return FileResponse(static_dir / "favicon.ico")

        @app.get("/manifest.json")
        async def manifest() -> FileResponse:
            return FileResponse(static_dir / "manifest.json")

        @app.get("/robots.txt")
        async def robots() -> FileResponse:
            return FileResponse(static_dir / "robots.txt")

        @app.get("/logo{size}.png")
        async def logo(size: int) -> FileResponse:
            return FileResponse(static_dir / f"logo{size}.png")

        # SPA fallback - serve index.html for non-API frontend routes
        # Register specific frontend routes instead of catch-all
        @app.get("/")
        async def spa_root():
            """Redirect to configuration page."""
            from fastapi.responses import RedirectResponse

            return RedirectResponse(url="/configuration")

        @app.get("/calibration")
        async def spa_calibration():
            """Serve index.html for calibration route."""
            return FileResponse(str(static_dir / "index.html"))

        @app.get("/configuration")
        async def spa_configuration():
            """Serve index.html for configuration route."""
            return FileResponse(str(static_dir / "index.html"))

        @app.get("/system-management")
        async def spa_system_management():
            """Serve index.html for system management route."""
            return FileResponse(str(static_dir / "index.html"))

        @app.get("/diagnostics")
        async def spa_diagnostics():
            """Serve index.html for diagnostics route."""
            return FileResponse(str(static_dir / "index.html"))

        # No else needed - if static dir doesn't exist, routes just won't be registered

    return app


# Dependency injection functions are now in dependencies module


# Create the application instance
app = create_app({"development_mode": True})
