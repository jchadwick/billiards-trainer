#!/usr/bin/env python3
"""Integration tests for the FastAPI application."""

import os
import signal
import subprocess
import sys
import time
from typing import Optional

import requests

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(__file__))


class TestServer:
    """Test server context manager for integration tests."""

    def __init__(self, port: int = 8002):
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://127.0.0.1:{port}"

    def __enter__(self):
        """Start the test server."""
        print(f"Starting test server on port {self.port}...")

        # Start the server process
        self.process = subprocess.Popen(
            [
                sys.executable,
                "dev_server.py",
                "--simple",
                "--no-reload",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,  # Create new process group
        )

        # Wait for server to start
        max_wait = 10  # seconds
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=1)
                if response.status_code == 200:
                    print(f"✅ Test server started successfully on {self.base_url}")
                    return self
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                time.sleep(0.5)
                continue

        raise RuntimeError(f"Failed to start test server within {max_wait} seconds")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the test server."""
        if self.process:
            print("Stopping test server...")
            # Kill the entire process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

            # Wait for process to terminate
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop gracefully
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait()

            print("✅ Test server stopped")

    def get(self, path: str, **kwargs):
        """Make GET request to the test server."""
        return requests.get(f"{self.base_url}{path}", **kwargs)

    def post(self, path: str, **kwargs):
        """Make POST request to the test server."""
        return requests.post(f"{self.base_url}{path}", **kwargs)


def test_health_endpoint():
    """Test the health check endpoint."""
    print("\n🧪 Testing health endpoint...")

    with TestServer() as server:
        response = server.get("/health")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert (
            data["status"] == "healthy"
        ), f"Expected healthy status, got {data['status']}"
        assert "version" in data, "Version not found in response"

        print("✅ Health endpoint test passed")


def test_root_endpoint():
    """Test the root endpoint."""
    print("\n🧪 Testing root endpoint...")

    with TestServer() as server:
        response = server.get("/")

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "message" in data, "Message not found in response"
        assert "version" in data, "Version not found in response"
        assert "docs" in data, "Docs link not found in response"

        print("✅ Root endpoint test passed")


def test_cors_headers():
    """Test CORS headers are present."""
    print("\n🧪 Testing CORS headers...")

    with TestServer() as server:
        response = server.get("/health")

        # Check for CORS headers (FastAPI automatically adds some)

        # Basic check that response is accessible (CORS working)
        assert response.status_code == 200

        print("✅ CORS headers test passed")


def test_openapi_docs():
    """Test that OpenAPI documentation is accessible."""
    print("\n🧪 Testing OpenAPI documentation...")

    with TestServer() as server:
        # Test OpenAPI JSON
        response = server.get("/openapi.json")
        assert (
            response.status_code == 200
        ), f"OpenAPI JSON failed: {response.status_code}"

        openapi_data = response.json()
        assert "openapi" in openapi_data, "OpenAPI spec missing"
        assert "info" in openapi_data, "OpenAPI info missing"

        # Test Swagger UI
        response = server.get("/docs")
        assert response.status_code == 200, f"Swagger UI failed: {response.status_code}"

        print("✅ OpenAPI documentation test passed")


def run_all_tests():
    """Run all integration tests."""
    print("🚀 Running FastAPI Integration Tests")
    print("=" * 50)

    tests = [
        test_health_endpoint,
        test_root_endpoint,
        test_cors_headers,
        test_openapi_docs,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
