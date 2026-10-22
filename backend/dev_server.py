#!/usr/bin/env python3
"""Retained FastAPI compatibility fixture; its server launcher is retired."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, NoReturn

# FastAPI inspects postponed endpoint annotations at runtime, so this import is
# required outside TYPE_CHECKING even though the name is only used as a type.
from fastapi import WebSocket  # noqa: TC002

# Load the selected snapshot before reading settings or importing app modules.
from backend.config import Config

Config.bootstrap_from_environment()

from backend.config import config

if TYPE_CHECKING:
    from fastapi import FastAPI

# Configure logging from config
log_level = getattr(logging, config.get("development.logging.level", "INFO"))
log_format = config.get(
    "development.logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logging.basicConfig(level=log_level, format=log_format)

logger = logging.getLogger(__name__)


def create_simple_app() -> FastAPI:
    """Create a FastAPI compatibility fixture for offline tests only."""
    import json
    from datetime import datetime

    from fastapi import FastAPI, WebSocketDisconnect
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
    cors_methods = config.get("development.cors.allow_methods", ["*"])
    cors_headers = config.get("development.cors.allow_headers", ["*"])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        # The single-user API has no cookie/session authentication contract.
        allow_credentials=False,
        allow_methods=cors_methods,
        allow_headers=cors_headers,
        # Browsers on the separate development frontend port need to read the
        # descriptor paired with the Rust preview JPEG body.
        expose_headers=["X-Frame-Preview-Descriptor", "X-Frame-Preview-Schema"],
    )

    # Basic health check
    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {
            "status": "healthy",
            "version": app_version,
            "service": "billiards-trainer-dev",
        }

    # Root endpoint
    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "message": "Billiards Trainer API (Development)",
            "version": app_version,
            "docs": docs_url,
            "health": "/health",
        }

    # Simple WebSocket endpoint for development/diagnostic purposes
    @app.websocket("/api/v1/websocket/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
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


def main() -> NoReturn:
    """Reject the retired Python development-server command."""
    print(
        "The Python development server has been retired. "
        "Use 'make run' for the Rust application or 'make run-api' for the "
        "Rust gateway without live capture.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
