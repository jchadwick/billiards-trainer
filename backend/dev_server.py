#!/usr/bin/env python3
"""Development server script for the Billiards Trainer API."""

import logging
import os
import sys

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def create_simple_app():
    """Create a simplified FastAPI application for development."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Billiards Trainer API (Dev)",
        description="Development version of the Billiards Trainer API",
        version="1.0.0-dev",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add basic CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Basic health check
    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "version": "1.0.0-dev",
            "service": "billiards-trainer-dev",
        }

    # Root endpoint
    @app.get("/")
    def root():
        return {
            "message": "Billiards Trainer API (Development)",
            "version": "1.0.0-dev",
            "docs": "/docs",
            "health": "/health",
        }

    return app


def run_dev_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = True,
    log_level: str = "info",
):
    """Run the development server with hot reload."""
    try:
        import uvicorn

        logger.info("Starting Billiards Trainer API development server...")
        logger.info(f"Server will be available at http://{host}:{port}")
        logger.info("API documentation available at http://localhost:8000/docs")

        # Create the app
        app = create_simple_app()

        # Run with uvicorn
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True,
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

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)

    except Exception as e:
        logger.error(f"❌ Failed to import/run full application: {e}")
        logger.info("Falling back to simplified development server...")
        run_dev_server()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Billiards Trainer API Development Server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
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
