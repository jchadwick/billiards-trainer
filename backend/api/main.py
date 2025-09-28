"""FastAPI application setup and configuration."""

import logging
import time
from contextlib import asynccontextmanager, suppress
from typing import Any, Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Internal imports
try:
    from ..config import ConfigurationModule
    from ..core import CoreModule, CoreModuleConfig
except ImportError:
    # If running from the backend directory directly
    from core import CoreModule, CoreModuleConfig

    from config import ConfigurationModule

from .dependencies import app_state
from .middleware.authentication import AuthenticationMiddleware
from .middleware.cors import CORSConfig, setup_cors_middleware
from .middleware.error_handler import ErrorHandlerConfig, setup_error_handling
from .middleware.logging import LoggingConfig, setup_logging_middleware
from .middleware.performance import PerformanceConfig, setup_performance_monitoring
from .middleware.rate_limit import RateLimitConfig, setup_rate_limiting
from .middleware.security import SecurityConfig, setup_security_headers
from .middleware.tracing import TracingConfig, setup_tracing_middleware
from .routes import auth, calibration, config, game, health
from .websocket import (
    initialize_websocket_system,
    shutdown_websocket_system,
    websocket_handler,
    websocket_manager,
)
from .websocket.endpoints import websocket_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_middleware_config(development_mode: bool = False) -> dict[str, Any]:
    """Get middleware configuration based on environment."""
    return {
        "cors": CORSConfig(
            allow_origins=(
                ["*"]
                if development_mode
                else ["http://localhost:3000", "http://localhost:3001"]
            ),
            allow_credentials=not development_mode,
            max_age=600,
        ),
        "rate_limit": RateLimitConfig(
            requests_per_minute=1000 if development_mode else 60,
            requests_per_hour=10000 if development_mode else 1000,
            burst_size=20 if development_mode else 10,
            enable_per_endpoint_limits=True,
        ),
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
        "security": SecurityConfig(
            development_mode=development_mode,
            enable_hsts=not development_mode,
            enable_csp=True,
            hide_server_header=True,
        ),
        "performance": PerformanceConfig(
            enable_monitoring=True,
            enable_system_metrics=True,
            slow_request_threshold_ms=500.0,
            excluded_paths=["/health", "/metrics"],
        ),
    }


# Application state is now imported from dependencies module


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    logger.info("Starting up Billiards Trainer API...")
    app_state.startup_time = time.time()

    try:
        # Initialize configuration module
        logger.info("Initializing configuration module...")
        app_state.config_module = ConfigurationModule()
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

        # Initialize WebSocket components (use global instances)
        logger.info("Initializing WebSocket components...")
        app_state.websocket_manager = websocket_manager
        app_state.websocket_handler = websocket_handler

        # Start WebSocket system services
        await initialize_websocket_system()

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
    development_mode = (
        config_override.get("development_mode", False)
        if config_override else False
    )

    # Get middleware configuration
    middleware_config = get_middleware_config(development_mode)

    # Add middleware in reverse order (last added = first executed)
    # This ensures proper execution order for request/response processing

    # 1. Performance monitoring (last - measures everything)
    setup_performance_monitoring(app, middleware_config["performance"])

    # 2. Security headers (second to last)
    setup_security_headers(app, middleware_config["security"])

    # 3. Request tracing and correlation IDs
    setup_tracing_middleware(app, middleware_config["tracing"])

    # 4. Request/response logging
    setup_logging_middleware(app, middleware_config["logging"])

    # 5. Error handling (handles errors from subsequent middleware)
    setup_error_handling(app, middleware_config["error_handler"])

    # 6. Rate limiting (before authentication to prevent abuse)
    setup_rate_limiting(app, middleware_config["rate_limit"])

    # 7. Authentication middleware (if enabled)
    if not development_mode:  # Skip auth in development
        app.add_middleware(AuthenticationMiddleware)

    # 8. Trusted host middleware (security)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=(
            ["localhost", "127.0.0.1", "0.0.0.0", "*"]
            if development_mode
            else ["localhost", "127.0.0.1"]
        ),
    )

    # 9. CORS (first - handles preflight requests)
    setup_cors_middleware(app, middleware_config["cors"], development_mode)

    # Middleware setup complete - all comprehensive middleware is now integrated

    # Include routers with API prefix
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(config.router, prefix="/api/v1")
    app.include_router(calibration.router, prefix="/api/v1")
    app.include_router(game.router, prefix="/api/v1")
    app.include_router(auth.router, prefix="/api/v1")

    # Include WebSocket management endpoints
    app.include_router(websocket_router, prefix="/api/v1/websocket")

    # Include the actual WebSocket endpoint at root level (no prefix)
    # For now, create a simple working WebSocket endpoint
    async def simple_websocket_endpoint(websocket: WebSocket):
        """Simple WebSocket endpoint for testing."""
        await websocket.accept()
        try:
            while True:
                # Wait for client message
                data = await websocket.receive_text()
                # Echo the message back
                await websocket.send_text(f"Echo: {data}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            with suppress(Exception):
                await websocket.close()

    app.add_websocket_route("/ws", simple_websocket_endpoint)

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "Billiards Trainer API",
            "version": "1.0.0",
            "status": "healthy" if app_state.is_healthy else "unhealthy",
            "docs": "/docs",
        }

    return app


# Dependency injection functions are now in dependencies module


# Create the application instance
app = create_app({"development_mode": True})
