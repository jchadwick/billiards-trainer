"""In-process FastAPI fixture for contracts around the retired Python API.

The supported application server is Rust. These fixtures assemble the old
FastAPI routing and middleware surface only so offline compatibility tests can
keep checking HTTP and WebSocket contracts without publishing a Python server
factory from ``backend.api.main``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api import main
from backend.api.dependencies import app_state
from backend.api.routes import (
    calibration,
)
from backend.api.routes import config as config_routes
from backend.api.routes import game, health, logs, stream, vision
from backend.api.websocket.endpoints import websocket_router


def create_compatibility_app(
    config_override: dict[str, Any] | None = None,
) -> FastAPI:
    """Assemble the legacy API as a test-only app using production components.

    Lifespan, configuration overlays, middleware setup, routers, and static
    fallback intentionally come from the same modules used by the retired
    factory. Keeping this glue under ``backend/tests`` prevents Uvicorn or other
    runtime tools from discovering a Python application factory.
    """
    overrides = config_override or {}
    app = FastAPI(
        title="Billiards Trainer API",
        description="RESTful API and WebSocket server for billiards training system",
        version="1.0.0",
        lifespan=main.lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.runtime_config = main._RuntimeConfiguration(main.config, overrides)

    development_mode = (
        overrides.get("development_mode", False)
        if overrides
        else os.getenv("ENVIRONMENT", "production").lower() == "development"
        or os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    )
    middleware_config = main.get_middleware_config(development_mode, overrides)

    # CORS must be registered before the remaining middleware for WebSockets.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Frame-Preview-Descriptor", "X-Frame-Preview-Schema"],
    )

    metrics_config = main.MetricsConfig(**middleware_config["metrics"])
    app.add_middleware(main.MetricsMiddleware, config=metrics_config)
    main.setup_tracing_middleware(app, middleware_config["tracing"])
    main.setup_logging_middleware(app, middleware_config["logging"])
    main.setup_error_handling(app, middleware_config["error_handler"])

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(config_routes.router, prefix="/api/v1")
    app.include_router(calibration.router, prefix="/api/v1/vision")
    app.include_router(vision.router, prefix="/api/v1/vision")
    app.include_router(game.router, prefix="/api/v1")
    app.include_router(stream.router, prefix="/api/v1")
    app.include_router(logs.router, prefix="/api/v1")
    app.include_router(websocket_router, prefix="/api/v1/websocket")

    @app.get("/api")
    async def api_root() -> dict[str, Any]:
        return {
            "message": "Billiards Trainer API",
            "version": "1.0.0",
            "status": "healthy" if app_state.is_healthy else "unhealthy",
            "docs": "/docs",
        }

    static_dir = Path(main.__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount(
            "/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets"
        )

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

        @app.get("/")
        async def spa_root() -> FileResponse:
            """Serve the existing frontend entry point for the root route."""
            return FileResponse(static_dir / "index.html")

        @app.get("/live")
        async def spa_live() -> FileResponse:
            """Serve the existing frontend entry point for live view."""
            return FileResponse(static_dir / "index.html")

        @app.get("/calibration")
        async def spa_calibration() -> FileResponse:
            """Serve the existing frontend entry point for calibration."""
            return FileResponse(static_dir / "index.html")

        @app.get("/configuration")
        async def spa_configuration() -> FileResponse:
            """Serve the existing frontend entry point for configuration."""
            return FileResponse(static_dir / "index.html")

        @app.get("/system-management")
        async def spa_system_management() -> FileResponse:
            """Serve the existing frontend entry point for system management."""
            return FileResponse(static_dir / "index.html")

        @app.get("/diagnostics")
        async def spa_diagnostics() -> FileResponse:
            """Serve the existing frontend entry point for diagnostics."""
            return FileResponse(static_dir / "index.html")

    return app
