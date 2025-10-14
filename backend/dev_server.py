#!/usr/bin/env python3
"""Development server script for the Billiards Trainer API."""

import logging
import os
import sys

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import config module before setting up logging
from backend.config.manager import ConfigurationModule

# Load configuration
config = ConfigurationModule()

# Configure logging from config
log_level = getattr(logging, config.get("development.logging.level", "INFO"))
log_format = config.get(
    "development.logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logging.basicConfig(level=log_level, format=log_format)

logger = logging.getLogger(__name__)


def create_simple_app():
    """Create a simplified FastAPI application for development."""
    import json
    from datetime import datetime

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware

    # Get app settings from config
    app_title = config.get("development.app.title", "Billiards Trainer API (Dev)")
    app_description = config.get(
        "development.app.description",
        "Development version of the Billiards Trainer API",
    )
    app_version = config.get("development.app.version", "1.0.0-dev")
    docs_url = config.get("development.app.docs_url", "/docs")
    redoc_url = config.get("development.app.redoc_url", "/redoc")

    app = FastAPI(
        title=app_title,
        description=app_description,
        version=app_version,
        docs_url=docs_url,
        redoc_url=redoc_url,
    )

    # Add basic CORS for development from config
    cors_origins = config.get("development.cors.allow_origins", ["*"])
    cors_credentials = config.get("development.cors.allow_credentials", False)
    cors_methods = config.get("development.cors.allow_methods", ["*"])
    cors_headers = config.get("development.cors.allow_headers", ["*"])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_credentials,
        allow_methods=cors_methods,
        allow_headers=cors_headers,
    )

    # Basic health check
    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "version": app_version,
            "service": "billiards-trainer-dev",
        }

    # Root endpoint
    @app.get("/")
    def root():
        return {
            "message": "Billiards Trainer API (Development)",
            "version": app_version,
            "docs": docs_url,
            "health": "/health",
        }

    # Simple WebSocket endpoint for development/diagnostic purposes
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Basic WebSocket endpoint for development - echoes messages and sends periodic test events."""
        await websocket.accept()

        # Send connection message
        await websocket.send_text(
            json.dumps(
                {
                    "type": "connection",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "data": {
                        "client_id": "dev-client",
                        "message": "Connected to development WebSocket",
                        "version": app_version,
                    },
                }
            )
        )

        logger.info("WebSocket client connected (development mode)")

        try:
            while True:
                # Wait for messages from the client
                message = await websocket.receive_text()

                # Parse and handle the message
                try:
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")

                    logger.debug(f"Received WebSocket message: {msg_type}")

                    # Echo back with acknowledgment
                    response = {
                        "type": f"{msg_type}_response",
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "data": {
                            "received": data,
                            "message": f"Development server received {msg_type} message",
                        },
                    }

                    # Send pong for ping messages
                    if msg_type == "ping":
                        response = {
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "data": data.get("data", {}),
                        }

                    await websocket.send_text(json.dumps(response))

                except json.JSONDecodeError:
                    # If message is not JSON, echo it back
                    await websocket.send_text(message)

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected (development mode)")

    return app


def run_dev_server(
    host: str = None,
    port: int = None,
    reload: bool = None,
    log_level: str = None,
):
    """Run the development server with hot reload."""
    try:
        import uvicorn

        # Use config defaults if not provided
        if host is None:
            host = config.get("development.server.host", "0.0.0.0")
        if port is None:
            port = config.get("development.server.port", 8000)
        if reload is None:
            reload = config.get("development.server.reload", True)
        if log_level is None:
            log_level = config.get("development.server.log_level", "info")

        access_log = config.get("development.server.access_log", True)

        logger.info("Starting Billiards Trainer API development server...")
        logger.info(f"Server will be available at http://{host}:{port}")
        logger.info(f"API documentation available at http://localhost:{port}/docs")

        # Create the app
        app = create_simple_app()

        # Run with uvicorn
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=access_log,
        )

    except ImportError:
        logger.error("uvicorn is required to run the development server")
        logger.error("Install it with: pip install uvicorn[standard]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start development server: {e}")
        sys.exit(1)


def run_production_app():
    """Attempt to run the full production application."""
    try:
        logger.info("Attempting to import full application...")

        # Try to import the full application
        from api.main import app

        logger.info("✅ Full application imported successfully!")
        logger.info("Starting production-like server...")

        import uvicorn

        # Use config for production settings
        host = config.get("api.server.host", "0.0.0.0")
        port = config.get("api.server.port", 8000)
        log_level = config.get("api.server.log_level", "info")
        access_log = config.get("development.server.access_log", True)

        uvicorn.run(
            app, host=host, port=port, log_level=log_level, access_log=access_log
        )

    except Exception as e:
        logger.error(f"❌ Failed to import/run full application: {e}")
        logger.info("Falling back to simplified development server...")
        run_dev_server()


if __name__ == "__main__":
    import argparse

    # Get defaults from config
    default_host = config.get("development.server.host", "0.0.0.0")
    default_port = config.get("development.server.port", 8000)
    default_log_level = config.get("development.server.log_level", "info")

    parser = argparse.ArgumentParser(
        description="Billiards Trainer API Development Server"
    )
    parser.add_argument("--host", default=default_host, help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=default_port, help="Port to bind to"
    )
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--log-level", default=default_log_level, help="Log level")
    parser.add_argument(
        "--simple", action="store_true", help="Use simple app instead of full app"
    )
    parser.add_argument(
        "--production", action="store_true", help="Try to run full production app"
    )

    args = parser.parse_args()

    if args.simple:
        logger.info("Starting simplified development server...")
        run_dev_server(
            host=args.host,
            port=args.port,
            reload=not args.no_reload,
            log_level=args.log_level,
        )
    elif args.production:
        run_production_app()
    else:
        # Default: try production first, fallback to simple
        run_production_app()
