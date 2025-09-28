#!/usr/bin/env python3
"""Test script for FastAPI application startup."""

import os
import sys

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from api.main import app

    print("✅ FastAPI application imported successfully!")

    # Test basic app properties
    print(f"   App title: {app.title}")
    print(f"   App version: {app.version}")
    print(f"   App description: {app.description}")

    # Check if routes are registered
    routes = list(app.routes)
    print(f"   Total routes: {len(routes)}")

    # List API routes
    api_routes = [
        route
        for route in routes
        if hasattr(route, "path") and route.path.startswith("/api")
    ]
    print(f"   API routes: {len(api_routes)}")
    for route in api_routes:
        if hasattr(route, "methods"):
            methods = ", ".join(route.methods)
            print(f"     {methods} {route.path}")

    print("\n✅ FastAPI application structure validated successfully!")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Application creation failed: {e}")
    sys.exit(1)
