#!/usr/bin/env python3
"""Test script for comprehensive health monitoring integration.

This script tests all aspects of the health monitoring system:
- Component health checks
- API metrics tracking
- WebSocket connection monitoring
- System resource monitoring
- Error handling and alerts
"""

import asyncio
import json
import logging
import time
from typing import Any

import aiohttp
import websockets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthMonitoringTester:
    """Comprehensive health monitoring system tester."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the tester.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
        self.ws_url = base_url.replace("http", "ws") + "/ws"
        self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def test_basic_health_endpoint(self) -> dict[str, Any]:
        """Test basic health endpoint functionality."""
        logger.info("Testing basic health endpoint...")

        try:
            async with self.session.get(f"{self.api_url}/health/") as resp:
                if resp.status != 200:
                    return {"success": False, "error": f"HTTP {resp.status}"}

                data = await resp.json()
                return {
                    "success": True,
                    "status": data.get("status"),
                    "uptime": data.get("uptime"),
                    "version": data.get("version"),
                    "timestamp": data.get("timestamp"),
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_detailed_health_endpoint(self) -> dict[str, Any]:
        """Test detailed health endpoint with component information."""
        logger.info("Testing detailed health endpoint...")

        try:
            params = {"include_details": "true", "include_metrics": "true"}
            async with self.session.get(
                f"{self.api_url}/health/", params=params
            ) as resp:
                if resp.status != 200:
                    return {"success": False, "error": f"HTTP {resp.status}"}

                data = await resp.json()
                components = data.get("components", {})
                metrics = data.get("metrics", {})

                return {
                    "success": True,
                    "components_count": len(components),
                    "components": list(components.keys()),
                    "has_metrics": metrics is not None,
                    "metrics_keys": list(metrics.keys()) if metrics else [],
                    "component_details": {
                        name: {
                            "status": comp.get("status"),
                            "message": comp.get("message"),
                            "uptime": comp.get("uptime"),
                            "error_count": comp.get("error_count", 0),
                        }
                        for name, comp in components.items()
                    },
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_metrics_endpoint(self) -> dict[str, Any]:
        """Test metrics endpoint functionality."""
        logger.info("Testing metrics endpoint...")

        try:
            async with self.session.get(f"{self.api_url}/health/metrics") as resp:
                if resp.status != 200:
                    return {"success": False, "error": f"HTTP {resp.status}"}

                data = await resp.json()
                return {
                    "success": True,
                    "cpu_usage": data.get("cpu_usage"),
                    "memory_usage": data.get("memory_usage"),
                    "disk_usage": data.get("disk_usage"),
                    "api_requests_per_second": data.get("api_requests_per_second"),
                    "websocket_connections": data.get("websocket_connections"),
                    "average_response_time": data.get("average_response_time"),
                    "network_io": data.get("network_io", {}),
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_api_metrics_tracking(self) -> dict[str, Any]:
        """Test API metrics tracking by making multiple requests."""
        logger.info("Testing API metrics tracking...")

        try:
            # Get initial metrics
            async with self.session.get(f"{self.api_url}/health/metrics") as resp:
                initial_data = await resp.json()

            initial_requests = initial_data.get("api_requests_per_second", 0)

            # Make several API requests
            request_count = 10
            start_time = time.time()

            tasks = []
            for _i in range(request_count):
                task = self.session.get(f"{self.api_url}/health/")
                tasks.append(task)

            # Execute requests concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Close all responses
            for resp in responses:
                if hasattr(resp, "close"):
                    resp.close()

            elapsed_time = time.time() - start_time

            # Wait a bit for metrics to update
            await asyncio.sleep(2)

            # Get updated metrics
            async with self.session.get(f"{self.api_url}/health/metrics") as resp:
                updated_data = await resp.json()

            updated_requests = updated_data.get("api_requests_per_second", 0)
            avg_response_time = updated_data.get("average_response_time", 0)

            return {
                "success": True,
                "requests_made": request_count,
                "elapsed_time": elapsed_time,
                "initial_rps": initial_requests,
                "updated_rps": updated_requests,
                "rps_increased": updated_requests > initial_requests,
                "avg_response_time": avg_response_time,
                "successful_responses": sum(
                    1 for r in responses if hasattr(r, "status") and r.status == 200
                ),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_websocket_health_monitoring(self) -> dict[str, Any]:
        """Test WebSocket connection health monitoring."""
        logger.info("Testing WebSocket health monitoring...")

        try:
            # Get initial WebSocket connection count
            async with self.session.get(f"{self.api_url}/health/metrics") as resp:
                initial_data = await resp.json()
            initial_connections = initial_data.get("websocket_connections", 0)

            # Connect to WebSocket
            websocket = await websockets.connect(self.ws_url)

            try:
                # Send a test message
                await websocket.send("Hello, WebSocket!")
                response = await websocket.recv()

                # Wait for metrics to update
                await asyncio.sleep(2)

                # Get updated metrics
                async with self.session.get(f"{self.api_url}/health/metrics") as resp:
                    updated_data = await resp.json()
                updated_connections = updated_data.get("websocket_connections", 0)

                return {
                    "success": True,
                    "connection_established": True,
                    "message_sent": True,
                    "response_received": response is not None,
                    "response_content": response,
                    "initial_connections": initial_connections,
                    "updated_connections": updated_connections,
                    "connection_tracked": updated_connections > initial_connections,
                }

            finally:
                await websocket.close()

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_component_health_details(self) -> dict[str, Any]:
        """Test detailed component health information."""
        logger.info("Testing component health details...")

        try:
            params = {"include_details": "true"}
            async with self.session.get(
                f"{self.api_url}/health/", params=params
            ) as resp:
                data = await resp.json()

            components = data.get("components", {})
            component_health = {}

            for name, component in components.items():
                component_health[name] = {
                    "status": component.get("status"),
                    "healthy": component.get("status") in ["healthy", "degraded"],
                    "has_message": bool(component.get("message")),
                    "has_uptime": component.get("uptime") is not None,
                    "error_count": component.get("error_count", 0),
                    "last_check": component.get("last_check"),
                }

            return {
                "success": True,
                "components_found": list(component_health.keys()),
                "healthy_components": [
                    name
                    for name, health in component_health.items()
                    if health["healthy"]
                ],
                "unhealthy_components": [
                    name
                    for name, health in component_health.items()
                    if not health["healthy"]
                ],
                "component_details": component_health,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_error_handling(self) -> dict[str, Any]:
        """Test error handling in health endpoints."""
        logger.info("Testing error handling...")

        try:
            # Test invalid endpoints
            test_results = {}

            # Test non-existent endpoint
            try:
                async with self.session.get(
                    f"{self.api_url}/health/nonexistent"
                ) as resp:
                    test_results["invalid_endpoint"] = {
                        "status_code": resp.status,
                        "expected_404": resp.status == 404,
                    }
            except Exception as e:
                test_results["invalid_endpoint"] = {"error": str(e)}

            # Test malformed parameters
            try:
                params = {"include_details": "invalid_value"}
                async with self.session.get(
                    f"{self.api_url}/health/", params=params
                ) as resp:
                    test_results["malformed_params"] = {
                        "status_code": resp.status,
                        "handles_gracefully": resp.status in [200, 400, 422],
                    }
            except Exception as e:
                test_results["malformed_params"] = {"error": str(e)}

            return {"success": True, "test_results": test_results}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def test_kubernetes_endpoints(self) -> dict[str, Any]:
        """Test Kubernetes-style health endpoints."""
        logger.info("Testing Kubernetes-style endpoints...")

        try:
            results = {}

            # Test liveness endpoint
            async with self.session.get(f"{self.api_url}/health/live") as resp:
                data = await resp.json()
                results["liveness"] = {
                    "status_code": resp.status,
                    "response_data": data,
                    "has_status": "status" in data,
                    "status_alive": data.get("status") == "alive",
                }

            # Test readiness endpoint
            async with self.session.get(f"{self.api_url}/health/ready") as resp:
                data = await resp.json()
                results["readiness"] = {
                    "status_code": resp.status,
                    "response_data": data,
                    "has_components": "components" in data,
                    "is_ready": resp.status == 200,
                }

            return {"success": True, "results": results}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def run_comprehensive_test(self) -> dict[str, Any]:
        """Run comprehensive health monitoring tests."""
        logger.info("Starting comprehensive health monitoring tests...")

        test_results = {}

        # Run all tests
        tests = [
            ("basic_health", self.test_basic_health_endpoint),
            ("detailed_health", self.test_detailed_health_endpoint),
            ("metrics", self.test_metrics_endpoint),
            ("api_metrics_tracking", self.test_api_metrics_tracking),
            ("websocket_monitoring", self.test_websocket_health_monitoring),
            ("component_health", self.test_component_health_details),
            ("error_handling", self.test_error_handling),
            ("kubernetes_endpoints", self.test_kubernetes_endpoints),
        ]

        for test_name, test_func in tests:
            logger.info(f"Running {test_name} test...")
            try:
                result = await test_func()
                test_results[test_name] = result
                if result.get("success"):
                    logger.info(f"✓ {test_name} test passed")
                else:
                    logger.error(f"✗ {test_name} test failed: {result.get('error')}")
            except Exception as e:
                logger.error(f"✗ {test_name} test error: {e}")
                test_results[test_name] = {"success": False, "error": str(e)}

        # Calculate overall results
        successful_tests = sum(
            1 for result in test_results.values() if result.get("success")
        )
        total_tests = len(test_results)

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": successful_tests / total_tests * 100,
            "all_passed": successful_tests == total_tests,
            "detailed_results": test_results,
        }


async def main():
    """Main test execution function."""
    logger.info("Starting health monitoring integration tests...")

    async with HealthMonitoringTester() as tester:
        results = await tester.run_comprehensive_test()

        # Print summary
        print("\n" + "=" * 80)
        print("HEALTH MONITORING TEST RESULTS")
        print("=" * 80)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Successful: {results['successful_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Overall Result: {'PASS' if results['all_passed'] else 'FAIL'}")
        print("=" * 80)

        # Print detailed results
        print("\nDETAILED TEST RESULTS:")
        print("-" * 80)
        for test_name, result in results["detailed_results"].items():
            status = "PASS" if result.get("success") else "FAIL"
            error = f" - {result.get('error')}" if not result.get("success") else ""
            print(f"{test_name:25} [{status}]{error}")

        # Print interesting metrics if available
        if results["detailed_results"].get("metrics", {}).get("success"):
            metrics = results["detailed_results"]["metrics"]
            print("\nSYSTEM METRICS:")
            print(f"  CPU Usage: {metrics.get('cpu_usage', 'N/A')}%")
            print(f"  Memory Usage: {metrics.get('memory_usage', 'N/A')}%")
            print(f"  Disk Usage: {metrics.get('disk_usage', 'N/A')}%")
            print(f"  API RPS: {metrics.get('api_requests_per_second', 'N/A')}")
            print(
                f"  WebSocket Connections: {metrics.get('websocket_connections', 'N/A')}"
            )
            print(
                f"  Avg Response Time: {metrics.get('average_response_time', 'N/A')}ms"
            )

        print("\n" + "=" * 80)

        # Save detailed results to file
        with open("/tmp/health_monitoring_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
            print("Detailed results saved to: /tmp/health_monitoring_test_results.json")

        return results["all_passed"]


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        exit(1)
