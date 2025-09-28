#!/usr/bin/env python3
"""Simple test script for FastAPI application startup without circular imports."""

import os
import sys

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from fastapi import FastAPI

    print("✅ FastAPI imported successfully!")

    # Create a simple app
    app = FastAPI(title="Test App", version="1.0.0")

    @app.get("/")
    def root():
        return {"message": "Hello World"}

    print("✅ Simple FastAPI app created successfully!")

    # Test the route
    routes = [route for route in app.routes if hasattr(route, "path")]
    print(f"✅ Routes configured: {len(routes)}")
    for route in routes:
        methods = getattr(route, "methods", [])
        print(f"   {route.path} - {list(methods)}")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
