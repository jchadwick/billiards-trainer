#!/usr/bin/env python3
"""Simple demonstration of the projector WebSocket client functionality."""

import asyncio
import json
import logging
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Basic WebSocket client implementation for demonstration
class SimpleProjectorClient:
    """Simplified WebSocket client for demonstration purposes."""

    def __init__(self, url: str = "ws://localhost:8000/ws"):
        self.url = url
        self.websocket = None
        self.connected = False
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "connection_time": None,
        }

    async def connect(self) -> bool:
        """Connect to WebSocket server."""
        try:
            import websockets

            logger.info(f"Connecting to {self.url}...")

            self.websocket = await websockets.connect(
                self.url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10,
            )

            self.connected = True
            self.stats["connection_time"] = time.time()

            logger.info("✓ Connected successfully!")
            return True

        except ImportError:
            logger.error(
                "✗ websockets library not available. Install with: pip install websockets"
            )
            return False
        except Exception as e:
            logger.error(f"✗ Connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected")

    async def send_message(self, message: dict) -> bool:
        """Send message to server."""
        if not self.connected:
            return False

        try:
            message_str = json.dumps(message)
            await self.websocket.send(message_str)
            self.stats["messages_sent"] += 1
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            return False

    async def subscribe(self, streams: list) -> bool:
        """Subscribe to data streams."""
        message = {"type": "subscribe", "data": {"streams": streams}}
        return await self.send_message(message)

    async def ping(self) -> bool:
        """Send ping message."""
        message = {"type": "ping", "data": {"timestamp": time.time()}}
        return await self.send_message(message)

    async def listen(self, duration: float = 10.0):
        """Listen for messages for specified duration."""
        if not self.connected:
            return

        logger.info(f"Listening for messages for {duration} seconds...")
        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                try:
                    # Wait for message with timeout
                    message_str = await asyncio.wait_for(
                        self.websocket.recv(), timeout=1.0
                    )

                    self.stats["messages_received"] += 1

                    try:
                        message = json.loads(message_str)
                        message_type = message.get("type", "unknown")

                        logger.info(
                            f"📨 Received {message_type}: {message.get('data', {})}"
                        )

                        # Handle specific message types
                        if message_type == "state":
                            self._handle_state_message(message.get("data", {}))
                        elif message_type == "trajectory":
                            self._handle_trajectory_message(message.get("data", {}))
                        elif message_type == "alert":
                            self._handle_alert_message(message.get("data", {}))
                        elif message_type == "pong":
                            logger.info("🏓 Pong received")

                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON received")

                except asyncio.TimeoutError:
                    # Normal timeout, continue listening
                    continue

        except Exception as e:
            logger.error(f"Listen error: {e}")

    def _handle_state_message(self, data: dict):
        """Handle game state message."""
        ball_count = data.get("ball_count", 0)
        balls = data.get("balls", [])
        cue = data.get("cue", {})

        logger.info(f"🎱 Game State: {ball_count} balls detected")
        if balls:
            for ball in balls[:3]:  # Show first 3 balls
                pos = ball.get("position", [0, 0])
                logger.info(f"   Ball {ball.get('id', '?')}: position {pos}")

        if cue.get("detected"):
            logger.info(f"🎯 Cue detected at angle {cue.get('angle', 0):.1f}°")

    def _handle_trajectory_message(self, data: dict):
        """Handle trajectory prediction message."""
        lines = data.get("lines", [])
        collisions = data.get("collisions", [])
        confidence = data.get("confidence", 0)

        logger.info(
            f"📈 Trajectory: {len(lines)} lines, {len(collisions)} collisions, {confidence:.2f} confidence"
        )

        if lines:
            first_line = lines[0]
            start = first_line.get("start", [0, 0])
            end = first_line.get("end", [0, 0])
            logger.info(f"   First line: {start} → {end}")

    def _handle_alert_message(self, data: dict):
        """Handle alert message."""
        level = data.get("level", "info")
        message = data.get("message", "")
        code = data.get("code", "")

        emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}.get(
            level, "📢"
        )
        logger.info(f"{emoji} Alert [{level.upper()}] {code}: {message}")

    def get_stats(self):
        """Get connection statistics."""
        uptime = (
            time.time() - self.stats["connection_time"]
            if self.stats["connection_time"]
            else 0
        )
        return {
            "connected": self.connected,
            "uptime_seconds": uptime,
            "messages_sent": self.stats["messages_sent"],
            "messages_received": self.stats["messages_received"],
        }


async def demo_basic_connection():
    """Demonstrate basic WebSocket connection."""
    logger.info("🔌 Testing basic WebSocket connection...")

    client = SimpleProjectorClient()

    # Test connection
    if not await client.connect():
        logger.error("❌ Could not establish connection")
        return False

    # Test ping
    logger.info("📡 Sending ping...")
    if await client.ping():
        logger.info("✓ Ping sent successfully")
    else:
        logger.error("✗ Ping failed")

    # Test subscription
    logger.info("📡 Subscribing to streams...")
    streams = ["state", "trajectory", "alert"]
    if await client.subscribe(streams):
        logger.info(f"✓ Subscribed to: {streams}")
    else:
        logger.error("✗ Subscription failed")

    # Listen for messages
    await client.listen(duration=5.0)

    # Show stats
    stats = client.get_stats()
    logger.info(f"📊 Stats: {stats}")

    # Disconnect
    await client.disconnect()

    return True


async def demo_message_simulation():
    """Demonstrate handling of simulated messages."""
    logger.info("🎭 Testing message handling with simulated data...")

    client = SimpleProjectorClient()

    # Simulate message handling without actual connection
    logger.info("Simulating game state message...")
    client._handle_state_message(
        {
            "ball_count": 3,
            "balls": [
                {"id": "cue", "position": [100, 200], "color": "white"},
                {"id": "1", "position": [300, 150], "color": "yellow"},
                {"id": "8", "position": [400, 300], "color": "black"},
            ],
            "cue": {"detected": True, "angle": 45.0, "position": [80, 180]},
        }
    )

    logger.info("Simulating trajectory message...")
    client._handle_trajectory_message(
        {
            "lines": [
                {"start": [100, 200], "end": [300, 150], "type": "primary"},
                {"start": [300, 150], "end": [400, 100], "type": "reflection"},
            ],
            "collisions": [{"position": [300, 150], "ball_id": "1", "angle": 30.0}],
            "confidence": 0.85,
            "calculation_time_ms": 12.5,
        }
    )

    logger.info("Simulating alert message...")
    client._handle_alert_message(
        {
            "level": "warning",
            "message": "Low confidence prediction",
            "code": "PRED_LOW_CONF",
            "details": {"confidence": 0.45},
        }
    )


async def check_server_availability():
    """Check if WebSocket server is available."""
    logger.info("🔍 Checking if WebSocket server is available...")

    try:
        import websockets

        # Try to connect briefly
        websocket = await asyncio.wait_for(
            websockets.connect("ws://localhost:8000/ws"), timeout=3.0
        )
        await websocket.close()

        logger.info("✅ WebSocket server is available at ws://localhost:8000/ws")
        return True

    except ImportError:
        logger.error("❌ websockets library not available")
        logger.error("   Install with: pip install websockets")
        return False
    except asyncio.TimeoutError:
        logger.error("❌ WebSocket server not responding (timeout)")
        logger.error("   Make sure the backend API server is running on port 8000")
        return False
    except Exception as e:
        logger.error(f"❌ WebSocket server not available: {e}")
        logger.error("   Start the server with: python backend/dev_server.py")
        return False


async def main():
    """Main demonstration function."""
    logger.info("🚀 Projector WebSocket Client Demo")
    logger.info("=" * 50)

    # Check server availability
    server_available = await check_server_availability()

    if server_available:
        logger.info("\n🟢 Server is available - running live connection demo")
        await demo_basic_connection()
    else:
        logger.info("\n🟡 Server not available - running simulation demo")
        await demo_message_simulation()

    logger.info("\n📋 WebSocket Client Features Demonstrated:")
    logger.info("   ✓ Connection establishment and management")
    logger.info("   ✓ Message subscription and handling")
    logger.info("   ✓ Game state processing")
    logger.info("   ✓ Trajectory data processing")
    logger.info("   ✓ Alert message handling")
    logger.info("   ✓ Performance statistics tracking")
    logger.info("   ✓ Error handling and recovery")

    logger.info("\n🎯 Next Steps:")
    logger.info("   1. Start the backend API server: python backend/dev_server.py")
    logger.info(
        "   2. Run the full projector with network: python backend/projector/main.py"
    )
    logger.info("   3. Connect the projector to receive real-time data")

    logger.info("\n✨ WebSocket Client Implementation Complete!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"\n💥 Demo failed: {e}")
        sys.exit(1)
