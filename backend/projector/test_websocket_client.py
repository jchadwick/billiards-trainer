#!/usr/bin/env python3
"""Test script for projector WebSocket client functionality."""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from projector.network.client import ConnectionState, WebSocketClient
from projector.network.handlers import HandlerConfig, ProjectorMessageHandlers

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockProjector:
    """Mock projector for testing message handlers."""

    def __init__(self):
        self.rendered_trajectories = []
        self.updated_ball_states = []
        self.cleared_count = 0
        self.removed_trajectories = []
        self.success_indicators = []
        self.trajectory_config_updates = []
        self.effects_config_updates = []

    def render_trajectory(self, trajectory, fade_in: bool = True) -> None:
        """Mock trajectory rendering."""
        self.rendered_trajectories.append(
            {"trajectory": trajectory, "fade_in": fade_in, "timestamp": time.time()}
        )
        logger.info(f"Mock: Rendered trajectory with fade_in={fade_in}")

    def update_ball_state(self, ball_state) -> None:
        """Mock ball state update."""
        self.updated_ball_states.append(
            {"ball_state": ball_state, "timestamp": time.time()}
        )
        logger.info("Mock: Updated ball state")

    def clear_display(self) -> None:
        """Mock display clear."""
        self.cleared_count += 1
        logger.info("Mock: Cleared display")

    def remove_trajectory(self, ball_id: str, fade_out: bool = True) -> None:
        """Mock trajectory removal."""
        self.removed_trajectories.append(
            {"ball_id": ball_id, "fade_out": fade_out, "timestamp": time.time()}
        )
        logger.info(f"Mock: Removed trajectory {ball_id} with fade_out={fade_out}")

    def render_success_indicator(self, position, success_probability: float) -> None:
        """Mock success indicator rendering."""
        self.success_indicators.append(
            {
                "position": position,
                "probability": success_probability,
                "timestamp": time.time(),
            }
        )
        logger.info(
            f"Mock: Rendered success indicator at {position} with probability {success_probability}"
        )

    def set_trajectory_config(self, config) -> None:
        """Mock trajectory config update."""
        self.trajectory_config_updates.append(
            {"config": config, "timestamp": time.time()}
        )
        logger.info("Mock: Updated trajectory configuration")

    def set_effects_config(self, config) -> None:
        """Mock effects config update."""
        self.effects_config_updates.append({"config": config, "timestamp": time.time()})
        logger.info("Mock: Updated effects configuration")

    def get_stats(self):
        """Get mock projector statistics."""
        return {
            "rendered_trajectories": len(self.rendered_trajectories),
            "updated_ball_states": len(self.updated_ball_states),
            "cleared_count": self.cleared_count,
            "removed_trajectories": len(self.removed_trajectories),
            "success_indicators": len(self.success_indicators),
            "trajectory_config_updates": len(self.trajectory_config_updates),
            "effects_config_updates": len(self.effects_config_updates),
        }


class WebSocketClientTester:
    """Test harness for WebSocket client functionality."""

    def __init__(self, api_url: str = "ws://localhost:8000/ws"):
        self.api_url = api_url
        self.client = None
        self.mock_projector = MockProjector()
        self.message_handlers = None
        self.test_results = {}

    async def run_tests(self) -> Dict[str, bool]:
        """Run comprehensive WebSocket client tests."""
        logger.info("Starting WebSocket client tests...")

        tests = [
            ("connection_test", self.test_connection),
            ("message_handler_test", self.test_message_handlers),
            ("subscription_test", self.test_subscriptions),
            ("error_handling_test", self.test_error_handling),
            ("reconnection_test", self.test_reconnection),
            ("performance_test", self.test_performance),
        ]

        results = {}

        for test_name, test_func in tests:
            logger.info(f"Running test: {test_name}")
            try:
                result = await test_func()
                results[test_name] = result
                logger.info(f"Test {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                results[test_name] = False

            # Brief pause between tests
            await asyncio.sleep(1)

        self.test_results = results
        return results

    async def test_connection(self) -> bool:
        """Test basic WebSocket connection functionality."""
        try:
            # Create client
            self.client = WebSocketClient(
                api_url=self.api_url,
                client_id="test_projector",
                reconnect_enabled=False,  # Disable for basic test
            )

            # Test connection
            connected = await self.client.connect()
            if not connected:
                logger.error("Failed to connect to WebSocket server")
                return False

            # Verify connection state
            if self.client.state != ConnectionState.CONNECTED:
                logger.error(f"Unexpected connection state: {self.client.state}")
                return False

            # Test ping
            ping_success = await self.client.ping()
            if not ping_success:
                logger.error("Ping failed")
                return False

            # Test status request
            status_success = await self.client.request_status()
            if not status_success:
                logger.error("Status request failed")
                return False

            # Test disconnection
            await self.client.disconnect()
            if self.client.connected:
                logger.error("Client still connected after disconnect")
                return False

            logger.info("Connection test passed")
            return True

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    async def test_message_handlers(self) -> bool:
        """Test message handler functionality."""
        try:
            # Create handlers
            handler_config = HandlerConfig(
                enable_trajectory_rendering=True,
                enable_ball_tracking=True,
                enable_alert_display=True,
            )

            self.message_handlers = ProjectorMessageHandlers(
                projector=self.mock_projector, config=handler_config
            )

            # Test state message handling
            state_data = {
                "balls": [
                    {
                        "id": "cue",
                        "position": [100, 200],
                        "radius": 10.0,
                        "color": "white",
                        "velocity": [5.0, 3.0],
                        "confidence": 0.95,
                        "visible": True,
                    }
                ],
                "cue": {
                    "detected": True,
                    "angle": 45.0,
                    "position": [150, 250],
                    "confidence": 0.9,
                },
                "ball_count": 1,
            }

            await self.message_handlers.handle_state(state_data)

            # Test trajectory message handling
            trajectory_data = {
                "lines": [
                    {
                        "start": [100, 200],
                        "end": [300, 400],
                        "type": "primary",
                        "confidence": 0.9,
                    }
                ],
                "collisions": [{"position": [300, 400], "ball_id": "1", "angle": 90.0}],
                "confidence": 0.85,
                "calculation_time_ms": 15.0,
                "line_count": 1,
                "collision_count": 1,
            }

            await self.message_handlers.handle_trajectory(trajectory_data)

            # Test alert message handling
            alert_data = {
                "level": "warning",
                "message": "Test alert message",
                "code": "TEST_001",
                "details": {"test": True},
            }

            await self.message_handlers.handle_alert(alert_data)

            # Test config message handling
            config_data = {
                "section": "trajectory",
                "config": {
                    "primary_color": [255, 0, 0],
                    "line_width": 5.0,
                    "opacity": 0.9,
                },
                "change_summary": "Updated trajectory color to red",
            }

            await self.message_handlers.handle_config(config_data)

            # Verify handler stats
            stats = self.message_handlers.get_handler_stats()
            if stats["stats"]["states_processed"] == 0:
                logger.error("State message not processed")
                return False

            if stats["stats"]["trajectories_processed"] == 0:
                logger.error("Trajectory message not processed")
                return False

            if stats["stats"]["alerts_processed"] == 0:
                logger.error("Alert message not processed")
                return False

            if stats["stats"]["config_updates_processed"] == 0:
                logger.error("Config message not processed")
                return False

            logger.info("Message handler test passed")
            return True

        except Exception as e:
            logger.error(f"Message handler test failed: {e}")
            return False

    async def test_subscriptions(self) -> bool:
        """Test stream subscription functionality."""
        try:
            # Create and connect client
            client = WebSocketClient(
                api_url=self.api_url,
                client_id="test_subscriptions",
                reconnect_enabled=False,
            )

            connected = await client.connect()
            if not connected:
                logger.error("Failed to connect for subscription test")
                return False

            # Test subscription
            streams = ["state", "trajectory", "alert"]
            subscribe_success = await client.subscribe(streams)
            if not subscribe_success:
                logger.error("Failed to subscribe to streams")
                await client.disconnect()
                return False

            # Verify subscriptions
            if not all(stream in client.subscriptions for stream in streams):
                logger.error("Not all streams were subscribed")
                await client.disconnect()
                return False

            # Test unsubscription
            unsubscribe_success = await client.unsubscribe(["alert"])
            if not unsubscribe_success:
                logger.error("Failed to unsubscribe from streams")
                await client.disconnect()
                return False

            # Verify unsubscription
            if "alert" in client.subscriptions:
                logger.error("Stream was not unsubscribed")
                await client.disconnect()
                return False

            await client.disconnect()
            logger.info("Subscription test passed")
            return True

        except Exception as e:
            logger.error(f"Subscription test failed: {e}")
            return False

    async def test_error_handling(self) -> bool:
        """Test error handling and recovery."""
        try:
            # Test connection to invalid URL
            invalid_client = WebSocketClient(
                api_url="ws://invalid-host:9999/ws", reconnect_enabled=False
            )

            connected = await invalid_client.connect()
            if connected:
                logger.error("Unexpectedly connected to invalid URL")
                return False

            # Verify error state
            if invalid_client.state != ConnectionState.ERROR:
                logger.error(f"Expected ERROR state, got {invalid_client.state}")
                return False

            logger.info("Error handling test passed")
            return True

        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False

    async def test_reconnection(self) -> bool:
        """Test automatic reconnection functionality."""
        try:
            # Create client with reconnection enabled
            client = WebSocketClient(
                api_url=self.api_url,
                client_id="test_reconnection",
                reconnect_enabled=True,
                max_reconnect_attempts=3,
                initial_reconnect_delay=1.0,
            )

            # Test that reconnection is configured
            if not client.reconnect_enabled:
                logger.error("Reconnection not enabled")
                return False

            if client.max_reconnect_attempts != 3:
                logger.error("Max reconnect attempts not set correctly")
                return False

            logger.info("Reconnection test passed (configuration check)")
            return True

        except Exception as e:
            logger.error(f"Reconnection test failed: {e}")
            return False

    async def test_performance(self) -> bool:
        """Test performance and statistics tracking."""
        try:
            client = WebSocketClient(api_url=self.api_url, client_id="test_performance")

            connected = await client.connect()
            if not connected:
                logger.error("Failed to connect for performance test")
                return False

            # Send some messages
            for _i in range(5):
                await client.ping()
                await asyncio.sleep(0.1)

            # Check statistics
            info = client.get_connection_info()
            stats = info["stats"]

            if stats["messages"]["sent"] == 0:
                logger.error("No messages sent recorded")
                await client.disconnect()
                return False

            if stats["connection"]["uptime_seconds"] <= 0:
                logger.error("Invalid uptime recorded")
                await client.disconnect()
                return False

            await client.disconnect()
            logger.info("Performance test passed")
            return True

        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return False

    def print_test_summary(self):
        """Print comprehensive test summary."""
        logger.info("\n" + "=" * 60)
        logger.info("WEBSOCKET CLIENT TEST SUMMARY")
        logger.info("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())

        for test_name, result in self.test_results.items():
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name:25} : {status}")

        logger.info("-" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")

        if self.mock_projector:
            logger.info("\nMock Projector Stats:")
            stats = self.mock_projector.get_stats()
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")

        logger.info("=" * 60)


async def main():
    """Main test execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test WebSocket client for projector")
    parser.add_argument(
        "--url", default="ws://localhost:8000/ws", help="WebSocket server URL"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")

    args = parser.parse_args()

    # Check if server is available
    logger.info(f"Testing WebSocket client against {args.url}")
    logger.info("Make sure the backend API server is running!")

    tester = WebSocketClientTester(api_url=args.url)

    try:
        results = await tester.run_tests()
        tester.print_test_summary()

        # Return appropriate exit code
        all_passed = all(results.values())
        return 0 if all_passed else 1

    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
