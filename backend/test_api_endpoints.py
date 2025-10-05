#!/usr/bin/env python3
"""Comprehensive API endpoint testing script."""

import sys
import time
from typing import Any, Optional

import requests

BASE_URL = "http://localhost:8000"


class EndpointTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        self.failed_tests = []

    def test_endpoint(
        self,
        method: str,
        path: str,
        expected_status: int = 200,
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
        description: str = "",
    ) -> dict[str, Any]:
        """Test a single endpoint and return results."""
        url = f"{self.base_url}{path}"
        start_time = time.time()

        try:
            if method.upper() == "GET":
                response = self.session.get(url, headers=headers)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, headers=headers)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, headers=headers)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response_time = time.time() - start_time

            result = {
                "method": method.upper(),
                "path": path,
                "url": url,
                "description": description,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "response_time": response_time,
                "success": response.status_code == expected_status,
                "headers": dict(response.headers),
                "content_length": len(response.content),
            }

            # Try to parse JSON response
            try:
                result["response_data"] = response.json()
            except:
                result["response_text"] = response.text[:500]  # Truncate long responses

            self.test_results.append(result)

            if not result["success"]:
                self.failed_tests.append(result)

            return result

        except Exception as e:
            result = {
                "method": method.upper(),
                "path": path,
                "url": url,
                "description": description,
                "error": str(e),
                "success": False,
                "response_time": time.time() - start_time,
            }
            self.test_results.append(result)
            self.failed_tests.append(result)
            return result

    def print_result(self, result: dict[str, Any]):
        """Print a formatted test result."""
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        method = result["method"]
        path = result["path"]

        if "error" in result:
            print(f"{status} {method} {path} - ERROR: {result['error']}")
        else:
            status_code = result["status_code"]
            response_time = result.get("response_time", 0) * 1000
            print(f"{status} {method} {path} - {status_code} ({response_time:.1f}ms)")

            if not result["success"]:
                expected = result["expected_status"]
                print(f"    Expected: {expected}, Got: {status_code}")

    def test_health_endpoints(self):
        """Test all health-related endpoints."""
        print("\n🏥 Testing Health Endpoints")
        print("=" * 50)

        # Basic health check
        self.test_endpoint("GET", "/api/v1/health/", description="Basic health check")

        # Version info
        self.test_endpoint(
            "GET", "/api/v1/health/version", description="Version information"
        )

        # Metrics
        self.test_endpoint(
            "GET", "/api/v1/health/metrics", description="Health metrics"
        )

        # Readiness check
        self.test_endpoint("GET", "/api/v1/health/ready", description="Readiness check")

        # Liveness check
        self.test_endpoint("GET", "/api/v1/health/live", description="Liveness check")

        for result in self.test_results[-5:]:
            self.print_result(result)

    def test_config_endpoints(self):
        """Test configuration endpoints."""
        print("\n⚙️  Testing Configuration Endpoints")
        print("=" * 50)

        # Get configuration
        self.test_endpoint("GET", "/api/v1/config/", description="Get configuration")

        # Get schema
        self.test_endpoint(
            "GET", "/api/v1/config/schema", description="Get configuration schema"
        )

        # Export configuration
        self.test_endpoint(
            "GET", "/api/v1/config/export", description="Export configuration"
        )

        for result in self.test_results[-3:]:
            self.print_result(result)

    def test_game_endpoints(self):
        """Test game-related endpoints."""
        print("\n🎮 Testing Game Endpoints")
        print("=" * 50)

        # Get game state
        self.test_endpoint("GET", "/api/v1/game/state", description="Get game state")

        # Get game history
        self.test_endpoint(
            "GET", "/api/v1/game/history", description="Get game history"
        )

        # Get game stats
        self.test_endpoint(
            "GET", "/api/v1/game/stats", description="Get game statistics"
        )

        for result in self.test_results[-3:]:
            self.print_result(result)

    def test_calibration_endpoints(self):
        """Test calibration endpoints."""
        print("\n📐 Testing Calibration Endpoints")
        print("=" * 50)

        # List calibration sessions
        self.test_endpoint(
            "GET", "/api/v1/calibration/", description="List calibration sessions"
        )

        # Start calibration (this might create a session)
        self.test_endpoint(
            "POST",
            "/api/v1/calibration",
            data={"type": "geometric"},
            description="Start calibration session",
        )

        for result in self.test_results[-2:]:
            self.print_result(result)

    def test_websocket_endpoints(self):
        """Test WebSocket management endpoints."""
        print("\n🔌 Testing WebSocket Endpoints")
        print("=" * 50)

        # WebSocket health
        self.test_endpoint(
            "GET", "/api/v1/websocket/health", description="WebSocket health"
        )

        # WebSocket connections
        self.test_endpoint(
            "GET",
            "/api/v1/websocket/connections",
            description="List WebSocket connections",
        )

        # WebSocket metrics
        self.test_endpoint(
            "GET", "/api/v1/websocket/metrics", description="WebSocket metrics"
        )

        for result in self.test_results[-3:]:
            self.print_result(result)

    def generate_report(self):
        """Generate a comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = total_tests - len(self.failed_tests)
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        print("\n" + "=" * 60)
        print("📊 API ENDPOINT TEST REPORT")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {len(self.failed_tests)}")
        print(f"Pass Rate: {pass_rate:.1f}%")

        if self.failed_tests:
            print("\n❌ Failed Tests:")
            for result in self.failed_tests:
                if "error" in result:
                    print(
                        f"  {result['method']} {result['path']} - ERROR: {result['error']}"
                    )
                else:
                    print(
                        f"  {result['method']} {result['path']} - {result['status_code']} (expected {result['expected_status']})"
                    )

        # Calculate average response time
        response_times = [
            r.get("response_time", 0) for r in self.test_results if "response_time" in r
        ]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times) * 1000
            print(f"\n⏱️  Average Response Time: {avg_response_time:.1f}ms")

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": len(self.failed_tests),
            "pass_rate": pass_rate,
            "failed_test_details": self.failed_tests,
            "all_results": self.test_results,
        }


def main():
    """Main test execution."""
    print("🚀 Starting Comprehensive API Endpoint Testing")
    print("=" * 60)

    tester = EndpointTester()

    # Test each category of endpoints
    tester.test_health_endpoints()
    tester.test_config_endpoints()
    tester.test_game_endpoints()
    tester.test_calibration_endpoints()
    tester.test_websocket_endpoints()

    # Generate final report
    report = tester.generate_report()

    # Exit with error code if any tests failed
    if report["failed_tests"] > 0:
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
