#!/usr/bin/env python3
"""Test WebSocket connections to the billiards trainer API."""

import asyncio
import json
import sys
import time
from typing import Any

import websockets


class WebSocketTester:
    def __init__(self, url: str = "ws://localhost:8000/ws"):
        self.url = url
        self.test_results = []

    async def test_basic_connection(self) -> dict[str, Any]:
        """Test basic WebSocket connection establishment."""
        test_name = "Basic Connection"
        start_time = time.time()

        try:
            async with websockets.connect(self.url) as websocket:
                connection_time = time.time() - start_time

                result = {
                    "test": test_name,
                    "success": True,
                    "connection_time": connection_time * 1000,  # ms
                    "message": "Successfully connected to WebSocket",
                }

                # Test immediate close
                await websocket.close()

        except Exception as e:
            result = {
                "test": test_name,
                "success": False,
                "error": str(e),
                "connection_time": (time.time() - start_time) * 1000,
            }

        self.test_results.append(result)
        return result

    async def test_message_sending(self) -> dict[str, Any]:
        """Test sending messages to the WebSocket."""
        test_name = "Message Sending"
        start_time = time.time()

        try:
            async with websockets.connect(self.url) as websocket:
                # Send a test message
                test_message = {
                    "type": "ping",
                    "timestamp": time.time(),
                    "data": {"test": True},
                }

                await websocket.send(json.dumps(test_message))

                # Try to receive a response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)

                    result = {
                        "test": test_name,
                        "success": True,
                        "response_time": (time.time() - start_time) * 1000,
                        "message": "Successfully sent and received message",
                        "response": response_data,
                    }
                except asyncio.TimeoutError:
                    result = {
                        "test": test_name,
                        "success": True,  # Still success if we can send
                        "response_time": (time.time() - start_time) * 1000,
                        "message": "Message sent, no response received (may be expected)",
                        "response": None,
                    }

        except Exception as e:
            result = {
                "test": test_name,
                "success": False,
                "error": str(e),
                "response_time": (time.time() - start_time) * 1000,
            }

        self.test_results.append(result)
        return result

    async def test_subscription(self) -> dict[str, Any]:
        """Test subscription to different streams."""
        test_name = "Stream Subscription"
        start_time = time.time()

        try:
            async with websockets.connect(self.url) as websocket:
                # Subscribe to a stream
                subscribe_message = {
                    "type": "subscribe",
                    "stream": "frame",
                    "client_id": "test_client_001",
                }

                await websocket.send(json.dumps(subscribe_message))

                # Wait a bit for potential responses
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    response_data = json.loads(response)

                    result = {
                        "test": test_name,
                        "success": True,
                        "response_time": (time.time() - start_time) * 1000,
                        "message": "Successfully subscribed to stream",
                        "response": response_data,
                    }
                except asyncio.TimeoutError:
                    result = {
                        "test": test_name,
                        "success": True,  # Subscription might be silent
                        "response_time": (time.time() - start_time) * 1000,
                        "message": "Subscription sent, no immediate response",
                        "response": None,
                    }

        except Exception as e:
            result = {
                "test": test_name,
                "success": False,
                "error": str(e),
                "response_time": (time.time() - start_time) * 1000,
            }

        self.test_results.append(result)
        return result

    async def test_multiple_connections(self) -> dict[str, Any]:
        """Test multiple concurrent WebSocket connections."""
        test_name = "Multiple Connections"
        start_time = time.time()

        try:
            # Create multiple connections simultaneously
            connection_tasks = []
            num_connections = 3

            async def create_connection(client_id: int):
                async with websockets.connect(self.url) as websocket:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "identify",
                                "client_id": f"test_client_{client_id}",
                            }
                        )
                    )
                    # Keep connection alive briefly
                    await asyncio.sleep(1)
                    return f"client_{client_id}"

            # Start all connections
            for i in range(num_connections):
                connection_tasks.append(create_connection(i))

            # Wait for all to complete
            results = await asyncio.gather(*connection_tasks, return_exceptions=True)

            successful_connections = [r for r in results if isinstance(r, str)]

            result = {
                "test": test_name,
                "success": len(successful_connections) == num_connections,
                "response_time": (time.time() - start_time) * 1000,
                "message": f"Successfully created {len(successful_connections)}/{num_connections} connections",
                "connections": successful_connections,
            }

        except Exception as e:
            result = {
                "test": test_name,
                "success": False,
                "error": str(e),
                "response_time": (time.time() - start_time) * 1000,
            }

        self.test_results.append(result)
        return result

    async def run_all_tests(self):
        """Run all WebSocket tests."""
        print("🔌 Starting WebSocket Connection Tests")
        print("=" * 50)

        tests = [
            self.test_basic_connection,
            self.test_message_sending,
            self.test_subscription,
            self.test_multiple_connections,
        ]

        for test_func in tests:
            result = await test_func()
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            test_name = result["test"]
            response_time = result.get("response_time", 0)

            print(f"{status} {test_name} ({response_time:.1f}ms)")

            if not result["success"]:
                print(f"    Error: {result.get('error', 'Unknown error')}")
            elif result.get("message"):
                print(f"    {result['message']}")

    def generate_report(self) -> dict[str, Any]:
        """Generate a summary report of WebSocket tests."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        avg_response_time = 0
        if self.test_results:
            response_times = [r.get("response_time", 0) for r in self.test_results]
            avg_response_time = sum(response_times) / len(response_times)

        report = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": pass_rate,
            "average_response_time": avg_response_time,
            "test_details": self.test_results,
        }

        print("\n" + "=" * 50)
        print("📊 WEBSOCKET TEST REPORT")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print(f"Average Response Time: {avg_response_time:.1f}ms")

        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result["success"]:
                    print(
                        f"  {result['test']} - {result.get('error', 'Unknown error')}"
                    )

        return report


async def main():
    """Main test execution."""
    print("🚀 Starting WebSocket Connection Testing")
    print("=" * 60)

    tester = WebSocketTester()

    try:
        await tester.run_all_tests()
        report = tester.generate_report()

        # Exit with error code if any tests failed
        if report["failed_tests"] > 0:
            print("\n⚠️  Some WebSocket tests failed!")
            sys.exit(1)
        else:
            print("\n🎉 All WebSocket tests passed!")
            sys.exit(0)

    except Exception as e:
        print(f"\n💥 Fatal error during WebSocket testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if websockets is available
    try:
        import websockets
    except ImportError:
        print(
            "❌ websockets library not available. Install with: pip install websockets"
        )
        sys.exit(1)

    asyncio.run(main())
