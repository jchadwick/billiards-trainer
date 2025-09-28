"""WebSocket client example for testing real-time data streaming functionality."""

import asyncio
import base64
import io
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import websockets
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClientStats:
    """Client connection statistics."""

    messages_received: int = 0
    messages_sent: int = 0
    frames_received: int = 0
    bytes_received: int = 0
    connection_time: Optional[float] = None
    last_ping_latency: float = 0.0
    average_latency: float = 0.0
    latency_measurements: list[float] = field(default_factory=list)


class BilliardsWebSocketClient:
    """Advanced WebSocket client for testing billiards trainer streaming."""

    def __init__(
        self, uri: str = "ws://localhost:8000/ws", token: Optional[str] = None
    ):
        self.uri = uri if not token else f"{uri}?token={token}"
        self.token = token
        self.websocket = None
        self.connected = False
        self.stats = ClientStats()
        self.subscriptions = set()

        # Event handlers
        self.frame_handler: Optional[Callable] = None
        self.state_handler: Optional[Callable] = None
        self.trajectory_handler: Optional[Callable] = None
        self.alert_handler: Optional[Callable] = None
        self.error_handler: Optional[Callable] = None

        # Configuration
        self.auto_reconnect = True
        self.reconnect_delay = 5.0
        self.ping_interval = 30.0

        # Tasks
        self.receive_task = None
        self.ping_task = None
        self.reconnect_task = None

    async def connect(self) -> bool:
        """Connect to the WebSocket server."""
        try:
            logger.info(f"Connecting to {self.uri}")
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            self.stats.connection_time = time.time()

            # Start tasks
            self.receive_task = asyncio.create_task(self._receive_loop())
            self.ping_task = asyncio.create_task(self._ping_loop())

            logger.info("Connected successfully")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        logger.info("Disconnecting...")
        self.connected = False

        # Cancel tasks
        if self.receive_task:
            self.receive_task.cancel()
        if self.ping_task:
            self.ping_task.cancel()
        if self.reconnect_task:
            self.reconnect_task.cancel()

        # Close connection
        if self.websocket:
            await self.websocket.close()

        logger.info("Disconnected")

    async def subscribe(self, streams: list[str]) -> bool:
        """Subscribe to data streams."""
        if not self.connected:
            logger.error("Not connected")
            return False

        message = {"type": "subscribe", "data": {"streams": streams}}

        success = await self._send_message(message)
        if success:
            self.subscriptions.update(streams)
            logger.info(f"Subscribed to streams: {streams}")

        return success

    async def unsubscribe(self, streams: list[str]) -> bool:
        """Unsubscribe from data streams."""
        if not self.connected:
            logger.error("Not connected")
            return False

        message = {"type": "unsubscribe", "data": {"streams": streams}}

        success = await self._send_message(message)
        if success:
            self.subscriptions.difference_update(streams)
            logger.info(f"Unsubscribed from streams: {streams}")

        return success

    async def get_status(self) -> bool:
        """Request connection status."""
        if not self.connected:
            logger.error("Not connected")
            return False

        message = {"type": "get_status", "data": {"include_details": True}}

        return await self._send_message(message)

    async def ping(self) -> float:
        """Send ping and measure latency."""
        if not self.connected:
            return -1.0

        start_time = time.time()
        message = {
            "type": "ping",
            "data": {"timestamp": datetime.now(timezone.utc).isoformat()},
        }

        success = await self._send_message(message)
        if not success:
            return -1.0

        # Note: This is a simplified latency measurement
        # Real latency would be measured when receiving the pong response
        return (time.time() - start_time) * 1000  # ms

    def set_frame_handler(self, handler: Callable[[dict[str, Any]], None]):
        """Set handler for frame messages."""
        self.frame_handler = handler

    def set_state_handler(self, handler: Callable[[dict[str, Any]], None]):
        """Set handler for game state messages."""
        self.state_handler = handler

    def set_trajectory_handler(self, handler: Callable[[dict[str, Any]], None]):
        """Set handler for trajectory messages."""
        self.trajectory_handler = handler

    def set_alert_handler(self, handler: Callable[[dict[str, Any]], None]):
        """Set handler for alert messages."""
        self.alert_handler = handler

    def set_error_handler(self, handler: Callable[[dict[str, Any]], None]):
        """Set handler for error messages."""
        self.error_handler = handler

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        uptime = (
            time.time() - self.stats.connection_time
            if self.stats.connection_time
            else 0
        )

        return {
            "connected": self.connected,
            "uptime_seconds": uptime,
            "messages_received": self.stats.messages_received,
            "messages_sent": self.stats.messages_sent,
            "frames_received": self.stats.frames_received,
            "bytes_received": self.stats.bytes_received,
            "subscriptions": list(self.subscriptions),
            "last_ping_latency_ms": self.stats.last_ping_latency,
            "average_latency_ms": self.stats.average_latency,
            "latency_measurements": len(self.stats.latency_measurements),
        }

    async def _send_message(self, message: dict[str, Any]) -> bool:
        """Send message to server."""
        if not self.connected or not self.websocket:
            return False

        try:
            message_str = json.dumps(message)
            await self.websocket.send(message_str)
            self.stats.messages_sent += 1
            return True

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    async def _receive_loop(self):
        """Main message receiving loop."""
        while self.connected:
            try:
                message_str = await self.websocket.recv()
                self.stats.messages_received += 1
                self.stats.bytes_received += len(message_str.encode())

                try:
                    message = json.loads(message_str)
                    await self._handle_message(message)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")

            except websockets.exceptions.ConnectionClosed:
                logger.warning("Connection closed by server")
                self.connected = False
                if self.auto_reconnect:
                    self.reconnect_task = asyncio.create_task(self._reconnect())
                break

            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                if self.error_handler:
                    await self._safe_call_handler(self.error_handler, {"error": str(e)})

    async def _handle_message(self, message: dict[str, Any]):
        """Handle incoming message based on type."""
        message_type = message.get("type")
        data = message.get("data", {})

        if message_type == "frame":
            self.stats.frames_received += 1
            if self.frame_handler:
                await self._safe_call_handler(self.frame_handler, data)

        elif message_type == "state":
            if self.state_handler:
                await self._safe_call_handler(self.state_handler, data)

        elif message_type == "trajectory":
            if self.trajectory_handler:
                await self._safe_call_handler(self.trajectory_handler, data)

        elif message_type == "alert":
            if self.alert_handler:
                await self._safe_call_handler(self.alert_handler, data)

        elif message_type == "error":
            logger.error(f"Server error: {data}")
            if self.error_handler:
                await self._safe_call_handler(self.error_handler, data)

        elif message_type == "pong":
            # Handle pong response for latency calculation
            timestamp_str = data.get("timestamp")
            if timestamp_str:
                try:
                    server_time = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    latency = (
                        datetime.now(timezone.utc) - server_time
                    ).total_seconds() * 1000
                    self.stats.last_ping_latency = latency
                    self.stats.latency_measurements.append(latency)

                    # Keep only recent measurements
                    if len(self.stats.latency_measurements) > 100:
                        self.stats.latency_measurements = (
                            self.stats.latency_measurements[-100:]
                        )

                    # Update average
                    self.stats.average_latency = sum(
                        self.stats.latency_measurements
                    ) / len(self.stats.latency_measurements)

                except Exception as e:
                    logger.warning(f"Failed to calculate latency: {e}")

        elif message_type == "connection":
            logger.info(f"Connection status: {data}")

        elif message_type == "subscribed":
            logger.info(f"Subscription confirmed: {data}")

        elif message_type == "unsubscribed":
            logger.info(f"Unsubscription confirmed: {data}")

        elif message_type == "status":
            logger.info(f"Status response: {data}")

        else:
            logger.debug(f"Unknown message type: {message_type}")

    async def _safe_call_handler(self, handler: Callable, data: dict[str, Any]):
        """Safely call a handler function."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(data)
            else:
                handler(data)
        except Exception as e:
            logger.error(f"Error in handler: {e}")

    async def _ping_loop(self):
        """Periodic ping loop."""
        while self.connected:
            try:
                await asyncio.sleep(self.ping_interval)
                if self.connected:
                    await self.ping()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping loop: {e}")

    async def _reconnect(self):
        """Attempt to reconnect."""
        while self.auto_reconnect and not self.connected:
            logger.info(f"Attempting to reconnect in {self.reconnect_delay} seconds...")
            await asyncio.sleep(self.reconnect_delay)

            if await self.connect():
                # Resubscribe to previous streams
                if self.subscriptions:
                    await self.subscribe(list(self.subscriptions))
                break


class InteractiveClient:
    """Interactive client for manual testing."""

    def __init__(self):
        self.client = BilliardsWebSocketClient()
        self.setup_handlers()

    def setup_handlers(self):
        """Setup message handlers."""
        self.client.set_frame_handler(self.handle_frame)
        self.client.set_state_handler(self.handle_state)
        self.client.set_trajectory_handler(self.handle_trajectory)
        self.client.set_alert_handler(self.handle_alert)
        self.client.set_error_handler(self.handle_error)

    def handle_frame(self, data: dict[str, Any]):
        """Handle frame data."""
        logger.info(
            f"Frame: {data['width']}x{data['height']}, {data['fps']:.1f} FPS, {data['size_bytes']} bytes"
        )

        # Optionally decode and display image
        if "image" in data:
            try:
                image_data = base64.b64decode(data["image"])
                image = Image.open(io.BytesIO(image_data))
                logger.info(f"Decoded image: {image.size}, {image.mode}")
                # You could save or display the image here
            except Exception as e:
                logger.warning(f"Failed to decode image: {e}")

    def handle_state(self, data: dict[str, Any]):
        """Handle game state data."""
        ball_count = data.get("ball_count", 0)
        cue_detected = data.get("cue", {}).get("detected", False)
        logger.info(
            f"Game state: {ball_count} balls, cue {'detected' if cue_detected else 'not detected'}"
        )

    def handle_trajectory(self, data: dict[str, Any]):
        """Handle trajectory data."""
        line_count = data.get("line_count", 0)
        collision_count = data.get("collision_count", 0)
        confidence = data.get("confidence", 0)
        logger.info(
            f"Trajectory: {line_count} lines, {collision_count} collisions, {confidence:.2f} confidence"
        )

    def handle_alert(self, data: dict[str, Any]):
        """Handle alert data."""
        level = data.get("level", "unknown")
        message = data.get("message", "No message")
        code = data.get("code", "NO_CODE")
        logger.warning(f"Alert [{level.upper()}] {code}: {message}")

    def handle_error(self, data: dict[str, Any]):
        """Handle error data."""
        code = data.get("code", "UNKNOWN")
        message = data.get("message", "Unknown error")
        logger.error(f"Error {code}: {message}")

    async def run_interactive(self):
        """Run interactive client session."""
        print("Billiards Trainer WebSocket Client")
        print("==================================")

        # Connect
        if not await self.client.connect():
            print("Failed to connect")
            return

        print("Connected! Available commands:")
        print("  sub <streams>  - Subscribe to streams (frame,state,trajectory,alert)")
        print("  unsub <streams> - Unsubscribe from streams")
        print("  ping          - Send ping")
        print("  status        - Get status")
        print("  stats         - Show client stats")
        print("  quit          - Quit")
        print()

        try:
            while self.client.connected:
                try:
                    command = input("> ").strip().lower()

                    if command == "quit":
                        break
                    elif command == "ping":
                        latency = await self.client.ping()
                        print(f"Ping: {latency:.2f} ms")
                    elif command == "status":
                        await self.client.get_status()
                    elif command == "stats":
                        stats = self.client.get_stats()
                        print(json.dumps(stats, indent=2))
                    elif command.startswith("sub "):
                        streams = command[4:].split(",")
                        streams = [s.strip() for s in streams if s.strip()]
                        await self.client.subscribe(streams)
                    elif command.startswith("unsub "):
                        streams = command[6:].split(",")
                        streams = [s.strip() for s in streams if s.strip()]
                        await self.client.unsubscribe(streams)
                    elif command == "help":
                        print(
                            "Available commands: sub, unsub, ping, status, stats, quit"
                        )
                    else:
                        print("Unknown command. Type 'help' for available commands.")

                except (EOFError, KeyboardInterrupt):
                    break
                except Exception as e:
                    print(f"Error: {e}")

        finally:
            await self.client.disconnect()


async def stress_test_client():
    """Stress test with multiple concurrent clients."""
    print("Starting stress test with multiple clients...")

    clients = []

    # Create multiple clients
    for i in range(10):
        client = BilliardsWebSocketClient(token=f"test_token_{i}")

        # Set up simple handlers
        client.set_frame_handler(
            lambda data, i=i: logger.info(f"Client {i} received frame")
        )
        client.set_alert_handler(
            lambda data, i=i: logger.info(
                f"Client {i} received alert: {data['message']}"
            )
        )

        clients.append(client)

    try:
        # Connect all clients
        tasks = [client.connect() for client in clients]
        results = await asyncio.gather(*tasks)

        connected_count = sum(results)
        print(f"Connected {connected_count}/{len(clients)} clients")

        # Subscribe all to frame stream
        for client in clients:
            if client.connected:
                await client.subscribe(["frame", "alert"])

        print("All clients subscribed. Running for 30 seconds...")
        await asyncio.sleep(30)

        # Show stats
        for i, client in enumerate(clients):
            if client.connected:
                stats = client.get_stats()
                print(
                    f"Client {i}: {stats['messages_received']} messages, {stats['frames_received']} frames"
                )

    finally:
        # Disconnect all clients
        tasks = [client.disconnect() for client in clients]
        await asyncio.gather(*tasks)


async def benchmark_client():
    """Benchmark client for performance testing."""
    print("Starting benchmark test...")

    client = BilliardsWebSocketClient()

    frame_count = 0
    start_time = None

    def frame_handler(data):
        nonlocal frame_count, start_time
        if start_time is None:
            start_time = time.time()
        frame_count += 1

        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"Received {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")

    client.set_frame_handler(frame_handler)

    try:
        if await client.connect():
            await client.subscribe(["frame"])
            print("Subscribed to frame stream. Benchmarking for 60 seconds...")
            await asyncio.sleep(60)

            elapsed = time.time() - start_time if start_time else 0
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Final: {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")

    finally:
        await client.disconnect()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "stress":
            asyncio.run(stress_test_client())
        elif mode == "benchmark":
            asyncio.run(benchmark_client())
        else:
            print("Usage: python client_example.py [stress|benchmark]")
            print("       python client_example.py  (for interactive mode)")
    else:
        # Interactive mode
        client = InteractiveClient()
        try:
            asyncio.run(client.run_interactive())
        except KeyboardInterrupt:
            print("\nGoodbye!")
